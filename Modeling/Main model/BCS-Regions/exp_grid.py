# =============================================================
# Nombre del archivo: exp_grid.py
#
# Descripción:
#   Experimento [19]: grid search del modelo actual (seg3 + DINOv2-L +
#   ridge, Dataset_Fr_BCS_v2, 64 vacas). No toca el modelo final.
#
#   Rejilla:
#     resolución de entrada  224 | 378 | 518 px (aplastada a cuadrado)
#     bloque de DINOv2-L     12 ... 24 (con LayerNorm final, como el modelo)
#     alpha del ridge        ALPHAS fijos + "interno" (elegido con
#                            GroupKFold(4) por vaca dentro de cada fold,
#                            exactamente como bcs_regions.fit_predict)
#   Validación: LOCO y 5 folds por vaca x 10 semillas (como el modelo).
#   Las máscaras son las de <work>/masks (seg3).
#
#   Pasos:
#     extract -> <work>/exp_grid/feats_<res>.npz   (GPU, ~6 min)
#     eval    -> <work>/exp_grid/grid.csv           (CPU)
#
# Uso: docker compose run --rm --entrypoint python bcs exp_grid.py extract
#      docker compose run --rm --entrypoint python bcs exp_grid.py eval
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br
from data import load_records
from segmenter import MEAN, STD

RES = [224, 378, 518]
BLOCKS = list(range(12, 25))                      # 1-based
ALPHAS = [1, 10, 100, 300, 1e3, 3e3, 1e4, 1e5]

@torch.no_grad()
def describe_blocks(bb, img, masks, size, dev):
    """-> [n_bloques, 6, 1024] promedios por región, [6] ausentes."""
    x = np.asarray(Image.fromarray(img).resize((size, size), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        outs = bb.forward_intermediates(x, indices=[b - 1 for b in BLOCKS], norm=True, output_fmt="NCHW", intermediates_only=True)
    g = size // 14
    m = F.interpolate(torch.from_numpy((masks > 0.5).astype(np.float32))[None], size=(size, size), mode="nearest")
    w = F.adaptive_avg_pool2d(m, (g, g))[0].reshape(6, -1).to(dev)
    tot = w.sum(1, keepdim=True); miss = tot[:, 0] < 0.5; regs = []
    for sp in outs:
        tok = sp[0].float().reshape(sp.shape[1], -1).T
        v = (w @ tok) / tot.clamp_min(1e-6); v[miss] = 0; regs.append(v.cpu().numpy())
    return np.stack(regs), miss.float().cpu().numpy()

def cmd_extract(a):
    work, dev = Path(a.work_dir), br.device(); out = work / "exp_grid"; out.mkdir(exist_ok=True)
    recs = load_records(a.data_dir)
    for size in RES:
        f = out / f"feats_{size}.npz"
        if f.exists(): print(f"[extract] {f} ya existe"); continue
        bb = timm.create_model(br.BACKBONE, pretrained=True, num_classes=0, img_size=size).to(dev).eval()
        R, M, t0 = [], [], time.time()
        for i, r in enumerate(recs):
            m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
            reg, miss = describe_blocks(bb, br.read_rgb(r["path"]), m, size, dev)
            R.append(reg.astype(np.float16)); M.append(miss)
            if (i + 1) % 100 == 0: print(f"  {size}px {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
        np.savez(f, reg=np.stack(R), miss=np.stack(M), y=np.array([r["bcs"] for r in recs]),
                 vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
        print(f"[extract] {size}px -> {f} ({time.time() - t0:.0f}s)", flush=True)
        del bb; torch.cuda.empty_cache()

def preds_all(X, y, cows, splits):
    """Una pasada por fold: predicciones para cada alpha fijo y para el alpha interno."""
    P = {al: np.zeros(len(y)) for al in ALPHAS}; P["interno"] = np.zeros(len(y)); chosen = []
    for tr, te in splits:
        A, B, _, _ = br.standardize(X[tr], X[te])
        for al, (p, _) in br.ridge_dual(A, B, y[tr], ALPHAS).items(): P[al][te] = p
        p, al = br.fit_predict(X[tr], y[tr], cows[tr], X[te]); P["interno"][te] = p; chosen.append(al)
    return P, chosen

def cmd_eval(a):
    out = Path(a.work_dir) / "exp_grid"; rows = []; t0 = time.time()
    for size in RES:
        z = np.load(out / f"feats_{size}.npz"); y, cows = z["y"], z["vaca"]; uc = np.unique(cows); strata = np.round(y * 2).astype(int)
        schemes = {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
                   "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(a.seeds)]}
        for j, blk in enumerate(BLOCKS):
            X = np.concatenate([z["reg"][:, j].astype(np.float64).reshape(len(y), -1), z["miss"]], 1)
            for sch, reps in schemes.items():
                acc = {}; ch = []
                for splits in reps:
                    P, c = preds_all(X, y, cows, splits); ch += c
                    for k, p in P.items(): acc.setdefault(k, []).append(p)
                for k, ps in acc.items():
                    m = br.metrics(np.mean(ps, 0), y, cows)
                    rows.append({"res": size, "bloque": blk, "alpha": k, "esquema": sch,
                                 **{q: round(float(v), 4) for q, v in m.items()},
                                 "alpha_elegido_moda": max(set(ch), key=ch.count) if k == "interno" else ""})
            best = [r for r in rows if r["res"] == size and r["bloque"] == blk and r["alpha"] == "interno"]
            print(f"  {size}px bloque {blk:2d} | " + " | ".join(f"{r['esquema']}: MAE={r['mae']:.3f} R2={r['r2_fotos']:.2f}" for r in best)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    with open(out / "grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"-> {out / 'grid.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "eval"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); {"extract": cmd_extract, "eval": cmd_eval}[a.step](a)

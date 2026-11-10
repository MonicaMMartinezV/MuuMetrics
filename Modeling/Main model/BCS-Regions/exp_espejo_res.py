# =============================================================
# Nombre del archivo: exp_espejo_res.py
#
# Descripción:
#   Experimento [26]: espejo y resolución con el modelo actual
#   (seg3 + DINOv3-L/16 + promedio por región + ridge, 64 vacas).
#   No toca el modelo final.
#
#   Rejilla:
#     resolución   320 | 384 (actual) | 448 | 512 px (aplastada a cuadrado)
#     espejo       0 = solo la foto (actual)
#                  1 = promedio de la foto y su espejo (foto y máscaras
#                      volteadas, muslo izq <-> der intercambiados; igual
#                      que exp_descriptores.py [4])
#     bloque       17 ... 24 (se guardan todos)
#
#   Análisis principal (sin selección): bloque 21 fijo, el del modelo
#   actual, para las 8 combinaciones resolución x espejo; LOCO y
#   rep5x10 con el mismo ridge del modelo (br.fit_predict) y diferencia
#   pareada de MAE por vaca contra la base 384 / sin espejo (IC95 bootstrap).
#   Análisis secundario (informativo, hay selección): barrido de bloques
#   con LOCO para cada combinación.
#
#   Pasos:  extract (GPU) -> <work>/exp_espejo_res/feats_<res>_<espejo>.npz
#           eval    (GPU/CPU) -> <work>/exp_espejo_res/resultados.csv,
#                               pareado.csv, barrido_bloques.csv, preds_loco.csv
# Uso: docker compose run --rm --entrypoint python bcs exp_espejo_res.py extract
#      docker compose run --rm --entrypoint python bcs exp_espejo_res.py eval
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

RES = [320, 384, 448, 512]
FLIPS = [0, 1]
BLOCKS = list(range(17, 25))      # 1-based
MAIN_BLOCK = br.LAYER + 1         # 21
SWAP = [0, 1, 2, 3, 5, 4]         # thigh_L <-> thigh_R
BASE = (384, 0)

@torch.no_grad()
def describe_blocks(bb, img, masks, size, flip, dev):
    """-> regiones [len(BLOCKS), 6, 1024], ausentes [6]"""
    if flip: img, masks = img[:, ::-1], masks[SWAP][:, :, ::-1]
    x = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((size, size), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - br.MEAN) / br.STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        outs = bb.forward_intermediates(x, indices=[b - 1 for b in BLOCKS], norm=True, output_fmt="NCHW", intermediates_only=True)
    g = outs[0].shape[-1]
    m = F.interpolate(torch.from_numpy((np.ascontiguousarray(masks) > 0.5).astype(np.float32))[None], size=(size, size), mode="nearest")
    w = F.adaptive_avg_pool2d(m, (g, g))[0].reshape(6, -1).to(dev)
    tot = w.sum(1, keepdim=True); miss = tot[:, 0] < 0.5; regs = []
    for sp in outs:
        tok = sp[0].float().reshape(sp.shape[1], -1).T
        v = (w @ tok) / tot.clamp_min(1e-6); v[miss] = 0; regs.append(v.cpu().numpy())
    return np.stack(regs), miss.float().cpu().numpy()

def cmd_extract(a):
    work, dev = Path(a.work_dir), br.device(); out = work / "exp_espejo_res"; out.mkdir(exist_ok=True)
    recs = load_records(a.data_dir)
    masks = [np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0 for r in recs]
    imgs = [br.read_rgb(r["path"]) for r in recs]
    meta = dict(y=np.array([r["bcs"] for r in recs]), vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
    for size in RES:
        todo = [f for f in FLIPS if not (out / f"feats_{size}_{f}.npz").exists()]
        if not todo: print(f"[extract] {size} ya existe"); continue
        bb = timm.create_model(br.BACKBONE, pretrained=True, num_classes=0, img_size=size).to(dev).eval()
        for flip in todo:
            R, M, t0 = [], [], time.time()
            for i in range(len(recs)):
                reg, miss = describe_blocks(bb, imgs[i], masks[i], size, flip, dev)
                R.append(reg.astype(np.float16)); M.append(miss)
                if (i + 1) % 100 == 0: print(f"  {size} espejo={flip} {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
            if dev.type == "cuda": torch.cuda.synchronize()
            secs = (time.time() - t0) / len(recs)
            np.savez(out / f"feats_{size}_{flip}.npz", reg=np.stack(R), miss=np.stack(M), blocks=np.array(BLOCKS), seg_por_foto=secs, **meta)
            print(f"[extract] {size} px espejo={flip}: malla {size // 16}x{size // 16}, {secs * 1000:.0f} ms/foto (8 bloques)", flush=True)
        del bb; torch.cuda.empty_cache()

# ---------------- evaluación ----------------
def build(z0, z1, j, flip):
    reg = z0["reg"][:, j].astype(np.float64)
    miss = z0["miss"]
    if flip:
        reg = (reg + z1["reg"][:, j].astype(np.float64)) / 2
        miss = np.maximum(miss, z1["miss"][:, SWAP])
    return np.concatenate([reg.reshape(len(reg), -1), miss], 1)

def predict(X, y, cows, splits_list):
    P = []
    for splits in splits_list:
        p = np.zeros(len(y))
        for tr, te in splits: p[te], _ = br.fit_predict(X[tr], y[tr], cows[tr], X[te])
        P.append(p)
    return np.mean(P, 0)

def cmd_eval(a):
    out = Path(a.work_dir) / "exp_espejo_res"; t0 = time.time()
    Z = {(s, f): np.load(out / f"feats_{s}_{f}.npz") for s in RES for f in FLIPS}
    z = Z[BASE]; y, cows, names = z["y"], z["vaca"], z["name"]; uc = np.unique(cows)
    for k in Z: assert (Z[k]["name"] == names).all(), k
    strata = np.round(y * 2).astype(int)
    S = {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
         "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(a.seeds)]}
    blocks = list(z["blocks"]); jm = blocks.index(MAIN_BLOCK)
    rows, preds = [], {}

    def label(s, f): return f"{s}px {'espejo' if f else 'sin espejo'}"

    print(f"[principal] bloque {MAIN_BLOCK} fijo, {len(y)} fotos, {len(uc)} vacas", flush=True)
    for s in RES:
        for f in FLIPS:
            X = build(Z[(s, 0)], Z[(s, 1)], jm, f); ms = float(Z[(s, 0)]["seg_por_foto"]) * (1 + f)
            for sch, spl in S.items():
                p = predict(X, y, cows, spl); m = br.metrics(p, y, cows); e = p - y
                rows.append({"variante": label(s, f), "res": s, "espejo": f, "bloque": MAIN_BLOCK, "esquema": sch,
                             **{k: round(float(v), 4) for k, v in m.items()},
                             "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3),
                             "ms_foto_8bloques": round(ms * 1000)})
                if sch == "loco": preds[label(s, f)] = p
                print(f"  {label(s, f):20s} {sch:7s} MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} R2v={m['r2_vacas']:.2f} ±0.5={m['acc_05']:.0%} "
                      f"MAE vaca={m['mae_vaca']:.3f} sesgo flacas={e[y < 3].mean():+.2f} gordas={e[y >= 4.5].mean():+.2f} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.RandomState(0); bl = label(*BASE)
    ref = np.array([np.abs(preds[bl] - y)[cows == u].mean() for u in uc])
    print(f"\nMAE por vaca vs {bl} (LOCO, bloque {MAIN_BLOCK}), IC95 bootstrap pareado:")
    with open(out / "pareado.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["variante", "dif", "ic_lo", "ic_hi", "mejoran", "vacas"])
        for k, p in preds.items():
            if k == bl: continue
            d = np.array([np.abs(p - y)[cows == u].mean() for u in uc]) - ref
            bs = [d[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
            w.writerow([k, round(d.mean(), 4), round(lo, 4), round(hi, 4), int((d < 0).sum()), len(uc)])
            print(f"  {k:20s} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  mejoran {(d < 0).sum()}/{len(uc)}")

    print(f"\n[secundario] barrido de bloques, LOCO (informativo: elegir aquí es selección)", flush=True)
    sweep = []
    for s in RES:
        for f in FLIPS:
            line = []
            for j, b in enumerate(blocks):
                p = predict(build(Z[(s, 0)], Z[(s, 1)], j, f), y, cows, S["loco"]); mae = np.abs(p - y).mean()
                sweep.append({"res": s, "espejo": f, "bloque": b, "mae_loco": round(float(mae), 4), "r2": round(float(br.r2(p, y)), 4)})
                line.append(f"{b}:{mae:.3f}")
            print(f"  {label(s, f):20s} " + " ".join(line), flush=True)

    with open(out / "resultados.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "barrido_bloques.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep[0])); w.writeheader(); w.writerows(sweep)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(preds))
        for i in range(len(y)): w.writerow([names[i], cows[i], y[i]] + [round(float(preds[k][i]), 4) for k in preds])
    print(f"-> {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "eval"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); {"extract": cmd_extract, "eval": cmd_eval}[a.step](a)

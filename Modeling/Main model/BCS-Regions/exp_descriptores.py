# =============================================================
# Nombre del archivo: exp_descriptores.py
#
# Descripción:
#   Experimento: mejores descriptores para el ridge de BCS.
#   Usa las máscaras ya calculadas (<work>/masks); no toca el modelo final.
#
#   Variantes que se extraen (una pasada por la GPU):
#     modo     squash  imagen aplastada a 378x378 (como el modelo base)
#              aspect  se conserva la proporción: alto x ancho múltiplos de
#                      14 con ~378² píxeles (DINOv2 con tamaño dinámico)
#     espejo   0/1     foto y máscaras volteadas horizontalmente, con
#                      muslo izq <-> der intercambiados
#     capas    bloques 12, 18 y 24 de DINOv2-L (24 = el que usa el base)
#     CLS      token global de cada capa
#
#   Pasos:
#     extract  -> <work>/exp_descriptores/feats.npz        (GPU, ~5 min)
#     eval     -> <work>/exp_descriptores/metrics.csv      (CPU)
#
# Uso: docker compose run --rm --entrypoint python bcs exp_descriptores.py extract
#      docker compose run --rm --entrypoint python bcs exp_descriptores.py eval
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

LAYERS = [11, 17, 23]            # índices 0-based de los bloques 12, 18, 24
MODES, FLIPS = ["squash", "aspect"], [0, 1]
SWAP = [0, 1, 2, 3, 5, 4]        # intercambia thigh_L <-> thigh_R

def input_size(h, w, mode):
    if mode == "squash": return br.SIZE, br.SIZE
    s = np.sqrt(br.SIZE ** 2 / (h * w))
    return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)

@torch.no_grad()
def describe_all(bb, img, masks, mode, flip, dev):
    """-> regiones [L,6,1024], cls [L,1024], ausentes [6]"""
    if flip: img, masks = img[:, ::-1], masks[SWAP][:, :, ::-1]
    H, W = input_size(*img.shape[:2], mode)
    x = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((W, H), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        outs = bb.forward_intermediates(x, indices=LAYERS, norm=True, return_prefix_tokens=True,
                                        output_fmt="NCHW", intermediates_only=True)
    gh, gw = H // 14, W // 14
    m = F.interpolate(torch.from_numpy((np.ascontiguousarray(masks) > 0.5).astype(np.float32))[None], size=(H, W), mode="nearest")
    w = F.adaptive_avg_pool2d(m, (gh, gw))[0].reshape(6, -1).to(dev)
    tot = w.sum(1, keepdim=True); miss = tot[:, 0] < 0.5
    regs, clss = [], []
    for sp, pre in outs:
        tok = sp[0].float().reshape(sp.shape[1], -1).T            # [gh*gw, 1024]
        v = (w @ tok) / tot.clamp_min(1e-6); v[miss] = 0
        regs.append(v.cpu().numpy()); clss.append(pre[0, 0].float().cpu().numpy())
    return np.stack(regs), np.stack(clss), miss.float().cpu().numpy()

def cmd_extract(a):
    work, dev = Path(a.work_dir), br.device(); recs = load_records(a.data_dir)
    bb = timm.create_model(br.BACKBONE, pretrained=True, num_classes=0, img_size=br.SIZE, dynamic_img_size=True).to(dev).eval()
    R = {(m, f): [] for m in MODES for f in FLIPS}; C = {k: [] for k in R}; M = {k: [] for k in R}; t0 = time.time()
    for i, r in enumerate(recs):
        img = br.read_rgb(r["path"]); masks = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
        for k in R:
            reg, cls, miss = describe_all(bb, img, masks, *k, dev)
            R[k].append(reg.astype(np.float16)); C[k].append(cls.astype(np.float16)); M[k].append(miss)
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    out = work / "exp_descriptores"; out.mkdir(exist_ok=True)
    arr = {}
    for (m, f) in R:
        arr[f"reg_{m}_{f}"] = np.stack(R[(m, f)]); arr[f"cls_{m}_{f}"] = np.stack(C[(m, f)]); arr[f"miss_{m}_{f}"] = np.stack(M[(m, f)])
    np.savez(out / "feats.npz", y=np.array([r["bcs"] for r in recs]), vaca=np.array([r["vaca"] for r in recs]),
             name=np.array([r["name"] for r in recs]), **arr)
    print(f"[extract] {len(recs)} fotos -> {out / 'feats.npz'} ({time.time() - t0:.0f}s)")

# ---------- Evaluación ----------
def build(z, mode, tta, layers, cls):
    """Concatena los bloques pedidos. layers: índices dentro de LAYERS (2 = último)."""
    def get(f):
        reg = z[f"reg_{mode}_{f}"][:, layers].astype(np.float32).reshape(len(z["y"]), -1)
        parts = [reg]
        if cls: parts.append(z[f"cls_{mode}_{f}"][:, layers].astype(np.float32).reshape(len(z["y"]), -1))
        return np.concatenate(parts, 1)
    X = (get(0) + get(1)) / 2 if tta else get(0)
    miss = np.maximum(z[f"miss_{mode}_0"], z[f"miss_{mode}_1"][:, SWAP]) if tta else z[f"miss_{mode}_0"]
    return np.concatenate([X, miss], 1).astype(np.float64)

VARIANTS = [  # nombre, modo, espejo, capas, CLS  (con 64 vacas, [18]: solo las que ayudaron en [4] y sus combinaciones)
    ("base (squash, capa 24)",        "squash", False, [2],       False),
    ("espejo",                        "squash", True,  [2],       False),
    ("capa 18",                       "squash", False, [1],       False),
    ("capa18+espejo",                 "squash", True,  [1],       False),
    ("capa18+aspect+espejo",          "aspect", True,  [1],       False),
    ("capas 18+24+espejo",            "squash", True,  [1, 2],    False),
]

def evaluate(X, y, cows, schemes):
    res = {}
    for sch, reps in schemes.items():
        P = []
        for splits in reps:
            p = np.zeros(len(y))
            for tr, te in splits: p[te], _ = br.fit_predict(X[tr], y[tr], cows[tr], X[te])
            P.append(p)
        res[sch] = np.mean(P, 0)
    return res

def cmd_eval(a):
    out = Path(a.work_dir) / "exp_descriptores"; z = dict(np.load(out / "feats.npz"))
    y, cows, names = z["y"], z["vaca"], z["name"]; uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(a.seeds)]}
    rows, loco = [], {}
    for name, mode, tta, layers, cls in VARIANTS:
        t0 = time.time(); X = build(z, mode, tta, layers, cls); res = evaluate(X, y, cows, schemes); loco[name] = res["loco"]
        line = []
        for sch, p in res.items():
            m = br.metrics(p, y, cows); e = p - y
            rows.append({"variante": name, "esquema": sch, "dims": X.shape[1], **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3)})
            line.append(f"{sch}: MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} ±0.5={m['acc_05']:.0%} MAEvaca={m['mae_vaca']:.3f}")
        print(f"{name:26s} dims={X.shape[1]:5d} | " + " | ".join(line) +
              f" | flacas={rows[-2]['sesgo_flacas']:+.2f} gordas={rows[-2]['sesgo_gordas']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco))
        for i in range(len(y)): w.writerow([names[i], cows[i], y[i]] + [round(float(loco[k][i]), 4) for k in loco])
    print(f"-> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["extract", "eval"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); {"extract": cmd_extract, "eval": cmd_eval}[a.step](a)

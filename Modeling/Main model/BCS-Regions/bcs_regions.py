# =============================================================
# Nombre del archivo: bcs_regions.py
#
# Descripción:
#   Modelo de BCS por regiones (modelo de referencia de model/main):
#     1. masks     segmentador -> 6 máscaras por foto (caché en <work>/masks)
#     2. features  DINOv2-L/14 congelado a 378 px; promedio de los tokens
#                  de parche dentro de cada región + indicador de región
#                  ausente -> vector de 6x1024 + 6 (caché <work>/features.npz)
#     3. evaluate  ridge validado por vaca: dejar una vaca fuera (LOCO) y
#                  5 folds por vaca x 10 semillas; alpha elegido con
#                  GroupKFold(4) por vaca dentro del train de cada fold
#     4. train     ridge final con todas las fotos -> <work>/ridge_model.npz
#     5. predict   fotos nuevas -> BCS estimado (csv)
#
# Uso: python bcs_regions.py <paso> [--data_dir ...] [--work_dir ...]
# =============================================================
import argparse, csv, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from data import load_records, IMG_EXTS
from segmenter import load_segmenter, segment, REGIONS, MEAN, STD

BACKBONE, SIZE = "vit_large_patch14_dinov2.lvd142m", 378
ALPHAS = (1, 10, 100, 1e3, 1e4, 1e5)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mask_file(work: Path, rec) -> Path:
    return work / "masks" / Path(rec["name"]).with_suffix(".npz")

def read_rgb(path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))

# ---------- 1. Máscaras ----------
def save_masks(path: Path, probs: np.ndarray):
    """Probabilidades [6,h,w] -> uint8 a media resolución (suficiente para la malla de 27x27)."""
    t = torch.from_numpy(probs)[None]
    m = F.interpolate(t, size=(probs.shape[1] // 2, probs.shape[2] // 2), mode="area")[0].numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, masks=(np.clip(m, 0, 1) * 255).round().astype(np.uint8), classes=np.array(REGIONS))

def cmd_masks(a):
    work, dev = Path(a.work_dir), device(); recs = load_records(a.data_dir)
    todo = [r for r in recs if a.overwrite or not mask_file(work, r).exists()]
    print(f"[masks] {len(recs) - len(todo)} ya existen, faltan {len(todo)}")
    if not todo: return
    net = load_segmenter(a.seg_weights, dev); empty = {c: 0 for c in REGIONS}
    for i, r in enumerate(todo):
        p = segment(net, read_rgb(r["path"]), dev); save_masks(mask_file(work, r), p)
        for c, v in zip(REGIONS, (p > 0.5).mean((1, 2))): empty[c] += int(v == 0)
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(todo)}", flush=True)
    print(f"[masks] listo. Fotos con una región vacía (fuera del recorte): {empty}")

# ---------- 2. Descriptores por región ----------
def load_backbone(dev):
    return timm.create_model(BACKBONE, pretrained=True, num_classes=0, img_size=SIZE).to(dev).eval()

@torch.no_grad()
def describe(bb, img: np.ndarray, masks: np.ndarray, dev) -> np.ndarray:
    """img RGB [h,w,3], masks [6,h',w'] en [0,1] -> [6*1024 promedios por región, 6 indicadores de ausencia]."""
    x = np.asarray(Image.fromarray(img).resize((SIZE, SIZE), Image.BICUBIC)) / 255.0   # aplasta a cuadrado
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        tok = bb.forward_features(x)
    patches = tok[0, bb.num_prefix_tokens:].float()                                  # [27*27, 1024]
    m = F.interpolate(torch.from_numpy((masks > 0.5).astype(np.float32))[None], size=(SIZE, SIZE), mode="nearest")
    w = F.adaptive_avg_pool2d(m, (SIZE // 14, SIZE // 14))[0].reshape(6, -1).to(dev)  # fracción de cada parche en la región
    tot = w.sum(1, keepdim=True); vec = (w @ patches) / tot.clamp_min(1e-6)
    miss = tot[:, 0] < 0.5; vec[miss] = 0                                             # menos de medio parche -> ausente
    return np.concatenate([vec.reshape(-1).cpu().numpy(), miss.float().cpu().numpy()])

def cmd_features(a):
    work, dev = Path(a.work_dir), device(); recs = load_records(a.data_dir)
    missing = [r["name"] for r in recs if not mask_file(work, r).exists()]
    if missing: raise SystemExit(f"Faltan {len(missing)} máscaras (p.ej. {missing[0]}): corre primero el paso 'masks'.")
    bb = load_backbone(dev); X = []
    for i, r in enumerate(recs):
        m = np.load(mask_file(work, r))["masks"].astype(np.float32) / 255.0
        X.append(describe(bb, read_rgb(r["path"]), m, dev))
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(recs)}", flush=True)
    np.savez_compressed(work / "features.npz", X=np.stack(X).astype(np.float32), y=np.array([r["bcs"] for r in recs]),
                        vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
    print(f"[features] {len(X)} x {len(X[0])} -> {work / 'features.npz'}")

# ---------- Ridge ----------
def standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    return (tr - mu) / sd, (te - mu) / sd, mu, sd

def ridge_dual(A, B, ytr, alphas):
    """Ridge con X ya estandarizada, resuelto con el kernel lineal (n << p). -> {alpha: (pred_B, coef)}"""
    ev, V = np.linalg.eigh(A @ A.T); yc = ytr - ytr.mean(); Vy = V.T @ yc; out = {}
    for al in alphas:
        dual = V @ (Vy / (ev + al)); out[al] = (B @ (A.T @ dual) + ytr.mean(), A.T @ dual)
    return out

def select_alpha(X, y, groups):
    err = {al: [] for al in ALPHAS}
    for i, v in GroupKFold(4).split(X, y, groups):
        A, B, _, _ = standardize(X[i], X[v])
        for al, (p, _) in ridge_dual(A, B, y[i], ALPHAS).items(): err[al].append(np.abs(p - y[v]))
    return min(ALPHAS, key=lambda al: np.mean(np.concatenate(err[al])))

def fit_predict(Xtr, ytr, gtr, Xte):
    al = select_alpha(Xtr, ytr, gtr); A, B, _, _ = standardize(Xtr, Xte)
    return ridge_dual(A, B, ytr, [al])[al][0], al

# ---------- 3. Evaluación ----------
def r2(p, y): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def metrics(p, y, cows):
    uc = np.unique(cows); cp = np.array([p[cows == c].mean() for c in uc]); cy = np.array([y[cows == c].mean() for c in uc])
    rd = np.clip(np.round(p / 0.25) * 0.25, 2.0, 5.0)
    per_cow = np.array([np.abs(p - y)[cows == c].mean() for c in uc])
    rng = np.random.RandomState(0); bs = [per_cow[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mae": np.abs(p - y).mean(), "rmse": np.sqrt(((p - y) ** 2).mean()), "r2_fotos": r2(p, y), "r2_vacas": r2(cp, cy),
            "acc_05": np.mean(np.abs(rd - y) <= 0.5 + 1e-9), "mae_vaca": per_cow.mean(), "mae_vaca_ic95_lo": lo, "mae_vaca_ic95_hi": hi}

def load_features(work: Path):
    f = work / "features.npz"
    if not f.exists(): raise SystemExit(f"No existe {f}: corre primero 'masks' y 'features'.")
    z = np.load(f); return z["X"].astype(np.float64), z["y"], z["vaca"], z["name"]

def cmd_evaluate(a):
    t0 = time.time(); work = Path(a.work_dir); X, y, cows, names = load_features(work); uc = np.unique(cows)
    strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    p0 = np.zeros(len(y))
    for tr, te in schemes["loco"][0]: p0[te] = y[tr].mean()
    rows = []
    for sch, reps in schemes.items():
        P, maes = [], []
        for splits in reps:
            p = np.zeros(len(y))
            for tr, te in splits: p[te], _ = fit_predict(X[tr], y[tr], cows[tr], X[te])
            P.append(p); maes.append(np.abs(p - y).mean())
        P = np.mean(P, 0); m = metrics(P, y, cows)
        rows.append({"esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                     "mae_reps_sd": round(float(np.std(maes)), 4), "fotos": len(y), "vacas": len(uc)})
        with open(work / f"preds_{sch}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
            w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, P)])
    with open(work / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{len(y)} fotos, {len(uc)} vacas | MAE prediciendo la media del train (LOCO): {np.abs(p0 - y).mean():.3f}")
    for r in rows:
        print(f"[{r['esquema']:7s}] MAE={r['mae']:.3f}  RMSE={r['rmse']:.3f}  R2 fotos={r['r2_fotos']:.2f}  R2 vacas={r['r2_vacas']:.2f}  "
              f"±0.5={r['acc_05']:.0%}  MAE por vaca={r['mae_vaca']:.3f} [{r['mae_vaca_ic95_lo']:.3f}, {r['mae_vaca_ic95_hi']:.3f}]")
    print(f"-> {work / 'metrics.csv'} ({time.time() - t0:.0f}s)")

# ---------- 4. Modelo final ----------
def cmd_train(a):
    work = Path(a.work_dir); X, y, cows, _ = load_features(work)
    al = select_alpha(X, y, cows); A, _, mu, sd = standardize(X, X[:1])
    coef = ridge_dual(A, A[:1], y, [al])[al][1] / sd                                   # coeficientes en el espacio original
    np.savez_compressed(work / "ridge_model.npz", coef=coef, intercept=y.mean() - coef @ mu, alpha=al,
                        meta=json.dumps({"backbone": BACKBONE, "size": SIZE, "regions": REGIONS, "fotos": len(y),
                                         "vacas": len(np.unique(cows))}))
    print(f"[train] alpha={al:g}, {len(y)} fotos -> {work / 'ridge_model.npz'}")

# ---------- 5. Predicción ----------
def cmd_predict(a):
    if not a.input: raise SystemExit("predict necesita --input <imagen o carpeta>")
    work, dev = Path(a.work_dir), device(); m = np.load(Path(a.model or work / "ridge_model.npz"))
    src = Path(a.input)
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    net, bb = load_segmenter(a.seg_weights, dev), load_backbone(dev); out = []
    for p in paths:
        img = read_rgb(p); x = describe(bb, img, segment(net, img, dev), dev)
        out.append([str(p.relative_to(src) if src.is_dir() else p.name), round(float(x @ m["coef"] + m["intercept"]), 3)])
    dst = Path(a.output or work / "predicciones.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "bcs_pred"]); w.writerows(out)
    for row in out[:10]: print(f"  {row[0]}: {row[1]}")
    print(f"[predict] {len(out)} fotos -> {dst}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Modelo de BCS por regiones: masks, features, evaluate, train, predict")
    ap.add_argument("step", choices=["masks", "features", "evaluate", "train", "predict", "all"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS", help="carpetas <vacaId>_<BCS>/")
    ap.add_argument("--work_dir", default="/work", help="caché de máscaras/features y resultados")
    ap.add_argument("--seg_weights", default="/weights/final_all40.pt")
    ap.add_argument("--seeds", type=int, default=10, help="repeticiones del esquema 5 folds por vaca")
    ap.add_argument("--overwrite", action="store_true", help="masks: recalcular aunque ya existan")
    ap.add_argument("--input", help="predict: imagen o carpeta de imágenes")
    ap.add_argument("--model", help="predict: ridge_model.npz (por defecto <work_dir>/ridge_model.npz)")
    ap.add_argument("--output", help="predict: csv de salida (por defecto <work_dir>/predicciones.csv)")
    a = ap.parse_args(); Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    steps = ["masks", "features", "evaluate", "train"] if a.step == "all" else [a.step]
    for s in steps: globals()[f"cmd_{s}"](a)

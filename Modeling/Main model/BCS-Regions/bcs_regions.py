# =============================================================
# Nombre del archivo: bcs_regions.py
#
# Descripción:
#   Modelo de BCS por regiones (modelo de referencia de model/main):
#     1. masks     segmentador -> 6 máscaras por foto (caché en <work>/masks)
#     2. features  DINOv2-L/14 congelado a 378 px; promedio de los tokens
#                  de parche dentro de cada región + indicador de región
#                  ausente -> vector de 6x1024 + 6 (caché <work>/features.npz)
#     3. evaluate  ridge + recalibración lineal, validado por vaca: dejar una
#                  vaca fuera (LOCO) y 5 folds por vaca x 10 semillas; alpha
#                  y la recta de calibración salen de un GroupKFold(4) por
#                  vaca dentro del train de cada fold. Dos líneas:
#                    todas     todas las fotos (modelo base)
#                    filtrada  solo fotos con lomo+cadera >= --min_lomo_cadera
#     4. train     modelo final de cada línea -> <work>/ridge_model.npz y
#                  <work>/ridge_model_filtrada.npz
#     5. predict   fotos nuevas -> BCS estimado + lomo+cadera visible (csv)
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

# ---------- Calibración y calidad ----------
def fit_model(X, y, groups):
    """Ridge + recalibración lineal. El GroupKFold(4) por vaca que elige alpha deja predicciones fuera de
    fold; con ellas se ajusta y ~ a + b*pred, que corrige parte del 'jalar al centro' del ridge."""
    oof = {al: np.zeros(len(y)) for al in ALPHAS}
    for i, v in GroupKFold(4).split(X, y, groups):
        A, B, _, _ = standardize(X[i], X[v])
        for al, (p, _) in ridge_dual(A, B, y[i], ALPHAS).items(): oof[al][v] = p
    al = min(ALPHAS, key=lambda a_: np.abs(oof[a_] - y).mean())
    A, _, mu, sd = standardize(X, X[:1]); coef = ridge_dual(A, A[:1], y, [al])[al][1] / sd   # coeficientes en el espacio original
    b, a = np.polyfit(oof[al], y, 1)
    return {"coef": coef, "intercept": y.mean() - coef @ mu, "alpha": al, "cal_a": a, "cal_b": b}

def apply_model(m, X):
    return np.clip(m["cal_a"] + m["cal_b"] * (X @ m["coef"] + m["intercept"]), 1, 5)

def lomo_cadera(masks: np.ndarray) -> float:
    """Fracción de la foto cubierta por lomo + cadera (masks [6,h,w] en [0,1])."""
    return float((masks[[REGIONS.index("loin"), REGIONS.index("hips")]] > 0.5).mean((1, 2)).sum())

def cached_quality(work: Path, names) -> np.ndarray:
    return np.array([lomo_cadera(np.load(work / "masks" / Path(n).with_suffix(".npz"))["masks"] / 255.0) for n in names])

# ---------- 3. Evaluación ----------
def r2(p, y): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def metrics(p, y, cows):
    uc = np.unique(cows); cp = np.array([p[cows == c].mean() for c in uc]); cy = np.array([y[cows == c].mean() for c in uc])
    rd = np.clip(np.round(p / 0.25) * 0.25, 2.0, 5.0)
    per_cow = np.array([np.abs(p - y)[cows == c].mean() for c in uc])
    rng = np.random.RandomState(0); bs = [per_cow[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); b = np.round(y * 2)
    return {"mae": np.abs(p - y).mean(), "rmse": np.sqrt(((p - y) ** 2).mean()), "r2_fotos": r2(p, y), "r2_vacas": r2(cp, cy),
            "acc_05": np.mean(np.abs(rd - y) <= 0.5 + 1e-9), "mae_vaca": per_cow.mean(), "mae_vaca_ic95_lo": lo, "mae_vaca_ic95_hi": hi,
            "mae_balanceado": np.mean([np.abs(p - y)[b == v].mean() for v in np.unique(b)]), "pendiente": np.polyfit(y, p, 1)[0]}

def load_features(work: Path):
    f = work / "features.npz"
    if not f.exists(): raise SystemExit(f"No existe {f}: corre primero 'masks' y 'features'.")
    z = np.load(f); return z["X"].astype(np.float64), z["y"], z["vaca"], z["name"]

def splits_for(y, cows, seeds):
    strata = np.round(y * 2).astype(int)
    return {"loco": [list(GroupKFold(len(np.unique(cows))).split(y, y, cows))],
            "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(seeds)]}

def cmd_evaluate(a):
    """Dos líneas: 'todas' (todas las fotos) y 'filtrada' (solo fotos con lomo+cadera >= --min_lomo_cadera,
    en train y en test). La línea 'todas' se reporta también sobre las fotos buenas, para comparar en las mismas fotos."""
    t0 = time.time(); work = Path(a.work_dir); X, y, cows, names = load_features(work)
    ok = cached_quality(work, names) >= a.min_lomo_cadera
    sin_foto = sorted(set(cows) - set(cows[ok]))
    print(f"{len(y)} fotos, {len(np.unique(cows))} vacas | lomo+cadera < {a.min_lomo_cadera}: {(~ok).sum()} fotos "
          f"de {len(set(cows[~ok]))} vacas; vacas sin ninguna foto evaluable: {sin_foto or 'ninguna'}")
    rows = []
    for linea, keep in (("todas", np.ones(len(y), bool)), ("filtrada", ok)):
        idx = np.where(keep)[0]; Xl, yl, cl, nl = X[idx], y[idx], cows[idx], names[idx]
        for sch, reps in splits_for(yl, cl, a.seeds).items():
            P, maes = [], []
            for splits in reps:
                p = np.zeros(len(yl))
                for tr, te in splits: p[te] = apply_model(fit_model(Xl[tr], yl[tr], cl[tr]), Xl[te])
                P.append(p); maes.append(np.abs(p - yl).mean())
            P = np.mean(P, 0)
            evals = [("todas", np.ones(len(yl), bool)), ("buenas", ok[idx])] if linea == "todas" else [("buenas", np.ones(len(yl), bool))]
            for ev, e in evals:
                m = metrics(P[e], yl[e], cl[e])
                rows.append({"linea": linea, "evaluado_en": ev, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                             "mae_reps_sd": round(float(np.std(maes)), 4), "fotos": int(e.sum()), "vacas": len(np.unique(cl[e]))})
            suf = "" if linea == "todas" else "_filtrada"
            with open(work / f"preds_{sch}{suf}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred", "evaluable"])
                w.writerows([[n, c, yy, round(float(pp), 4), "si" if ok[i] else "no"] for n, c, yy, pp, i in zip(nl, cl, yl, P, idx)])
    with open(work / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'línea':9s} {'evaluado':8s} {'esquema':8s} {'MAE':>6s} {'R2fot':>6s} {'R2vac':>6s} {'±0.5':>5s} {'pend':>5s} {'MAEvaca [IC95]':>22s} {'fotos':>5s} {'vacas':>5s}")
    for r in rows:
        print(f"{r['linea']:9s} {r['evaluado_en']:8s} {r['esquema']:8s} {r['mae']:6.3f} {r['r2_fotos']:6.3f} {r['r2_vacas']:6.3f} {r['acc_05']:5.0%} "
              f"{r['pendiente']:5.2f} {r['mae_vaca']:.3f} [{r['mae_vaca_ic95_lo']:.3f}, {r['mae_vaca_ic95_hi']:.3f}] {r['fotos']:5d} {r['vacas']:5d}")
    print(f"-> {work / 'metrics.csv'} ({time.time() - t0:.0f}s)")

# ---------- 4. Modelo final ----------
def cmd_train(a):
    """Entrena las dos líneas: ridge_model.npz (todas las fotos) y ridge_model_filtrada.npz."""
    work = Path(a.work_dir); X, y, cows, names = load_features(work)
    ok = cached_quality(work, names) >= a.min_lomo_cadera
    for suf, keep in (("", np.ones(len(y), bool)), ("_filtrada", ok)):
        m = fit_model(X[keep], y[keep], cows[keep]); dst = work / f"ridge_model{suf}.npz"
        np.savez_compressed(dst, **m, min_lomo_cadera=a.min_lomo_cadera,
                            meta=json.dumps({"backbone": BACKBONE, "size": SIZE, "regions": REGIONS, "fotos": int(keep.sum()),
                                             "vacas": len(np.unique(cows[keep])), "filtrada": bool(suf)}))
        print(f"[train] {dst.name}: alpha={m['alpha']:g}, calibración {m['cal_a']:+.3f} + {m['cal_b']:.3f}*pred, {keep.sum()} fotos")

# ---------- 5. Predicción ----------
def cmd_predict(a):
    """Cada foto: BCS estimado + fracción de lomo+cadera visible y si pasa el umbral. Es informativo: en la
    validación, promediar solo las fotos que pasan no mejoró el BCS por vaca (0.492 vs 0.482 con todas)."""
    if not a.input: raise SystemExit("predict necesita --input <imagen o carpeta>")
    work, dev = Path(a.work_dir), device(); m = dict(np.load(Path(a.model or work / "ridge_model.npz")))
    umbral = float(m.get("min_lomo_cadera", a.min_lomo_cadera))
    src = Path(a.input)
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    net, bb = load_segmenter(a.seg_weights, dev), load_backbone(dev); out = []
    for p in paths:
        img = read_rgb(p); masks = segment(net, img, dev); q = lomo_cadera(masks)
        bcs = float(apply_model(m, describe(bb, img, masks, dev)[None])[0])
        out.append([str(p.relative_to(src) if src.is_dir() else p.name), round(bcs, 3), round(q, 3), "si" if q >= umbral else "no"])
    dst = Path(a.output or work / "predicciones.csv")
    with open(dst, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "bcs_pred", "lomo_cadera", "evaluable"]); w.writerows(out)
    for row in out[:10]: print(f"  {row[0]}: {row[1]} ({'evaluable' if row[3] == 'si' else 'NO evaluable'})")
    ev = sum(r[3] == "si" for r in out)
    print(f"[predict] {len(out)} fotos ({ev} con lomo+cadera suficiente) -> {dst} | BCS promedio: {np.mean([r[1] for r in out]):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Modelo de BCS por regiones: masks, features, evaluate, train, predict")
    ap.add_argument("step", choices=["masks", "features", "evaluate", "train", "predict", "all"])
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS", help="carpetas <vacaId>_<BCS>/")
    ap.add_argument("--work_dir", default="/work", help="caché de máscaras/features y resultados")
    ap.add_argument("--seg_weights", default="/weights/final_all40.pt")
    ap.add_argument("--seeds", type=int, default=10, help="repeticiones del esquema 5 folds por vaca")
    ap.add_argument("--min_lomo_cadera", type=float, default=0.13,
                    help="fracción mínima de la foto con lomo+cadera para que sea evaluable (≈ percentil 10 de Dataset_Fr_BCS)")
    ap.add_argument("--overwrite", action="store_true", help="masks: recalcular aunque ya existan")
    ap.add_argument("--input", help="predict: imagen o carpeta de imágenes")
    ap.add_argument("--model", help="predict: ridge_model.npz (por defecto <work_dir>/ridge_model.npz)")
    ap.add_argument("--output", help="predict: csv de salida (por defecto <work_dir>/predicciones.csv)")
    a = ap.parse_args(); Path(a.work_dir).mkdir(parents=True, exist_ok=True)
    steps = ["masks", "features", "evaluate", "train"] if a.step == "all" else [a.step]
    for s in steps: globals()[f"cmd_{s}"](a)

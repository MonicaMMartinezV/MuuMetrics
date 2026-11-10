# =============================================================
# Nombre del archivo: exp_depth.py
#
# Descripción:
#   Experimento [14]: profundidad monocular con Depth Anything V2 (Large)
#   como información de relieve para el BCS. Usa las máscaras ya calculadas
#   (<work>/masks) y los descriptores DINOv2-L del modelo base
#   (<work>/features.npz); no toca el modelo final.
#
#   Por foto (y su espejo, promediados):
#     - mapa de profundidad relativa (DA-V2 da disparidad afín: escala y
#       desplazamiento arbitrarios) -> se normaliza con la mediana y el
#       rango intercuartil dentro de la unión de las 6 regiones;
#     - estadísticas por región: media, sd, p10, p90, pendiente media,
#       curvatura (laplaciano) media y sd  -> 6 x 7
#       + desniveles entre regiones (media de cada una menos la del lomo) -> 5
#     - tokens del encoder de DA-V2 (DINOv2-L ajustado a profundidad),
#       promediados por región como en el modelo base -> 6 x 1024
#
#   Pasos:
#     extract -> <work>/exp_depth/feats.npz + viz/ (GPU, ~3 min)
#     eval    -> <work>/exp_depth/metrics.csv, preds_loco.csv (CPU)
#
# Uso: ./run_depth.sh extract | eval
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br
from data import load_records

MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
REG = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
SWAP = [0, 1, 2, 3, 5, 4]
STATS = ["media", "sd", "p10", "p90", "pendiente", "curv", "curv_sd"]
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

def input_size(h, w, short=518):
    """Lado corto 518 (como el procesador de DA-V2), ambos lados múltiplos de 14."""
    s = short / min(h, w)
    return max(14, int(round(h * s / 14)) * 14), max(14, int(round(w * s / 14)) * 14)

@torch.no_grad()
def run_da(model, img, size, dev):
    """img RGB uint8 -> (disparidad [h,w] en size=(h,w) (la de las máscaras), tokens [gh*gw,1024], (gh,gw))."""
    H, W = input_size(*img.shape[:2]); h, w = size
    x = np.asarray(Image.fromarray(img).resize((W, H), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        out = model(pixel_values=x, output_hidden_states=True)
    d = F.interpolate(out.predicted_depth[:, None].float(), size=(h, w), mode="bicubic", align_corners=False)[0, 0]
    tok = out.hidden_states[-1][0, 1:].float()          # último bloque del encoder, sin CLS
    return d.cpu().numpy(), tok, (H // 14, W // 14)

def region_tokens(tok, grid, masks, dev):
    gh, gw = grid
    m = F.interpolate(torch.from_numpy((masks > 0.5).astype(np.float32))[None], size=(gh * 14, gw * 14), mode="nearest")
    wt = F.adaptive_avg_pool2d(m, (gh, gw))[0].reshape(6, -1).to(dev)
    tot = wt.sum(1, keepdim=True); v = (wt @ tok) / tot.clamp_min(1e-6); miss = tot[:, 0] < 0.5; v[miss] = 0
    return v.cpu().numpy(), miss.float().cpu().numpy()

def depth_stats(d, masks):
    """d: disparidad [h,w] (más cerca = mayor), masks [6,h,w] en [0,1] -> 6x7 + 5 desniveles (NaN si falta la región)."""
    M = masks > 0.5; U = M.any(0)
    if U.sum() < 50: return np.full(6 * 7 + 5, np.nan)
    med = np.median(d[U]); q1, q3 = np.percentile(d[U], [25, 75]); z = (d - med) / max(q3 - q1, 1e-6)
    gy, gx = np.gradient(z); grad = np.hypot(gx, gy)
    lap = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    out, means = [], []
    for k in range(6):
        if M[k].sum() < 30: out += [np.nan] * 7; means.append(np.nan); continue
        v = z[M[k]]; out += [v.mean(), v.std(), *np.percentile(v, [10, 90]), grad[M[k]].mean(), lap[M[k]].mean(), lap[M[k]].std()]
        means.append(v.mean())
    out += [means[k] - means[0] for k in range(1, 6)]
    return np.array(out, np.float32)

def colorize(d, U):
    v = d[U] if U.any() else d.ravel(); lo, hi = np.percentile(v, [2, 98])
    t = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([t, 1 - np.abs(2 * t - 1), 1 - t], -1)      # azul lejos -> rojo cerca
    return (rgb * 255).astype(np.uint8)

def preprocess(img, masks, mode):
    """Realce de iluminación antes de Depth Anything. clahe: contraste local en L (LAB);
    gamma: gamma por foto para que el brillo medio dentro de la vaca sea 0.5."""
    if mode == "none": return img
    import cv2
    if mode == "clahe":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    U = cv2.resize(((masks > 0.5).any(0)).astype(np.uint8), img.shape[1::-1], interpolation=cv2.INTER_NEAREST) > 0
    lum = img.astype(np.float32).mean(2) / 255; mu = np.clip(lum[U].mean() if U.any() else lum.mean(), 0.02, 0.98)
    g = np.log(0.5) / np.log(mu)
    return (np.clip((img / 255.0) ** g, 0, 1) * 255).astype(np.uint8)

def cmd_extract(a):
    from transformers import AutoModelForDepthEstimation
    work, dev = Path(a.work_dir), br.device(); out = work / ("exp_depth" + ("" if a.pre == "none" else "_" + a.pre)); (out / "viz").mkdir(parents=True, exist_ok=True)
    recs = load_records(a.data_dir)
    if a.limit: recs = recs[:a.limit]
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(dev).eval()
    S, T, MS, t0 = [], [], [], time.time(); viz_cows = set()
    for i, r in enumerate(recs):
        img = br.read_rgb(r["path"]); m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
        img = preprocess(img, m, a.pre)
        sz = m.shape[1:]; img_s = np.asarray(Image.fromarray(img).resize(sz[::-1], Image.BILINEAR))   # solo para la visualización
        d0, tok0, g0 = run_da(model, img, sz, dev); d1, tok1, g1 = run_da(model, np.ascontiguousarray(img[:, ::-1]), sz, dev)
        d = (d0 + d1[:, ::-1]) / 2                                            # TTA espejo
        v0, miss = region_tokens(tok0, g0, m, dev); v1, _ = region_tokens(tok1, g1, np.ascontiguousarray(m[SWAP][:, :, ::-1]), dev)
        S.append(depth_stats(d, m)); T.append(((v0 + v1) / 2).astype(np.float16)); MS.append(miss)
        cow = r["name"].split("/")[0]
        if len(viz_cows) < 8 and cow not in viz_cows:
            viz_cows.add(cow); U = (m > 0.5).any(0); c = colorize(d, U)
            edge = np.zeros_like(U)
            for k in range(6):
                b = m[k] > 0.5; edge |= b ^ np.pad(b, 1, mode="edge")[1:-1, 2:]
            c[edge] = 255
            Image.fromarray(np.concatenate([img_s, c], 1)).save(out / "viz" / f"{cow}.jpg", quality=90)
        if (i + 1) % 50 == 0: print(f"  {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out / "feats.npz", stats=np.stack(S), tok=np.stack(T), miss=np.stack(MS),
             y=np.array([r["bcs"] for r in recs]), vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
    print(f"[extract] {len(recs)} fotos -> {out / 'feats.npz'} ({time.time() - t0:.0f}s); viz -> {out / 'viz'}")

# ---------- Evaluación (mismo ridge, mismos esquemas que el modelo base) ----------
def stats_matrix(S):
    """NaN (región ausente) -> media de la columna + indicador por región."""
    S = S.astype(np.float64); nan = np.isnan(S); col = np.nanmean(S, 0); S[nan] = np.take(col, np.where(nan)[1])
    return S

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

def cow_diff(p, q, y, cows):
    """MAE por vaca de p menos el de q, IC95 bootstrap por vaca."""
    uc = np.unique(cows); d = np.array([np.abs(p - y)[cows == c].mean() - np.abs(q - y)[cows == c].mean() for c in uc])
    rng = np.random.RandomState(0); bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5]), int((d < 0).sum()), len(d)

def cmd_eval(a):
    work = Path(a.work_dir); out = work / ("exp_depth" + ("" if a.pre == "none" else "_" + a.pre)); z = dict(np.load(out / "feats.npz"))
    fb = np.load(work / "features.npz"); assert list(fb["name"]) == list(z["name"]), "orden de fotos distinto"
    y, cows, names = z["y"], z["vaca"], z["name"]; uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(a.seeds)]}
    Xb = fb["X"].astype(np.float64); S = stats_matrix(z["stats"]); miss = z["miss"].astype(np.float64)
    Td = np.concatenate([z["tok"].astype(np.float64).reshape(len(y), -1), miss], 1)
    # Los bloques se pesan para aportar la misma varianza total que el bloque DINOv2 (el ridge estandariza cada columna).
    wS = np.sqrt(Xb.shape[1] / (S.shape[1] + 6))
    V = {"base (DINOv2-L, modelo actual)": Xb,
         "relieve DA (48)": np.concatenate([S, miss], 1),
         "tokens DA (6x1024)": Td,
         "base + relieve": np.concatenate([Xb, S], 1),
         "base + relieve (bloques igual peso)": np.concatenate([Xb, np.concatenate([S, miss], 1) * wS], 1),
         "base + tokens DA": np.concatenate([Xb, Td[:, :-6]], 1)}
    rows, loco = [], {}
    for name, X in V.items():
        t0 = time.time(); res = evaluate(X, y, cows, schemes); loco[name] = res["loco"]; line = []
        for sch, p in res.items():
            m = br.metrics(p, y, cows); e = p - y
            rows.append({"variante": name, "esquema": sch, "dims": X.shape[1], **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3)})
            line.append(f"{sch}: MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} R2v={m['r2_vacas']:.2f} MAEvaca={m['mae_vaca']:.3f}")
        print(f"{name:36s} dims={X.shape[1]:5d} | " + " | ".join(line) +
              f" | flacas={rows[-2]['sesgo_flacas']:+.2f} gordas={rows[-2]['sesgo_gordas']:+.2f} ({time.time() - t0:.0f}s)", flush=True)
    base = loco["base (DINOv2-L, modelo actual)"]
    print("\nMAE por vaca vs base (LOCO, IC95 bootstrap por vaca):")
    for name, p in loco.items():
        if p is base: continue
        d, lo, hi, nb, n = cow_diff(p, base, y, cows); print(f"  {name:36s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]  mejoran {nb}/{n}")
    # Correlación de cada estadística de relieve con el BCS (por vaca), para ver si la profundidad trae señal
    from scipy.stats import spearmanr
    cols = [f"{r}_{s}" for r in REG for s in STATS] + [f"{r}-loin" for r in REG[1:]]
    Sv = np.array([S[cows == c].mean(0) for c in uc]); yv = np.array([y[cows == c].mean() for c in uc])
    rho = [(cols[j], *spearmanr(Sv[:, j], yv)) for j in range(len(cols))]; rho.sort(key=lambda t: t[2])
    print("\nRelieve más relacionado con el BCS (por vaca, Spearman, sin corregir por comparaciones múltiples):")
    for c, r_, p_ in rho[:8]: print(f"  {c:22s} rho={r_:+.2f} p={p_:.4f}")
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
    ap.add_argument("--pre", default="none", choices=["none", "clahe", "gamma"], help="realce de iluminación antes de DA")
    ap.add_argument("--limit", type=int, default=0, help="extract: solo las primeras N fotos (prueba)")
    a = ap.parse_args(); {"extract": cmd_extract, "eval": cmd_eval}[a.step](a)

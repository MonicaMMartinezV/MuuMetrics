# =============================================================
# Nombre del archivo: mask_bcs_corr.py
#
# Descripción:
#   Correlación entre características geométricas de las 6 máscaras
#   del segmentador y el BCS. Usa solo las máscaras en caché
#   (<work>/masks/*.npz, paso 'masks'): numpy + scipy + sklearn, sin GPU.
#
#   Características por foto (máscara binaria = prob > 0.5):
#     - por región: área relativa al recorte, proporción dentro de la
#       unión de las 6 regiones (invariante a escala), aspecto del bbox,
#       solidez (área / casco convexo), compacidad 4πA/P², confianza
#       (prob. media dentro de la región) e incertidumbre del borde
#     - globales: tamaño y aspecto del recorte, área/solidez de la unión,
#       simetría de muslos, traslapes entre regiones vecinas
#
#   Estadística (las fotos de una vaca NO son independientes):
#     - Spearman por foto con IC95 por bootstrap de vacas
#     - Spearman por animal-estado (promedio por vaca y BCS) con p por
#       permutación y corrección FDR (Benjamini-Hochberg)
#     - ICC(1): qué tanto de la variación es entre vacas y no por pose
#     - ridge LOCO solo con estas características (vs. DINOv2: MAE 0.583)
#
# Uso: python3 mask_bcs_corr.py [--data_dir ...] [--work_dir work] [--out work/analysis]
# =============================================================
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_records

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
NOMBRES = dict(zip(REGIONS, ["lomo", "cadera", "grupa", "base cola", "muslo izq", "muslo der"]))
PAIRS = [("hips", "rump"), ("rump", "tailhead"), ("rump", "thigh_L"), ("rump", "thigh_R"), ("loin", "hips")]

# ---------- Características ----------
def perimeter(m):
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] != m[:-1, :]; e[:-1, :] |= m[:-1, :] != m[1:, :]
    e[:, 1:] |= m[:, 1:] != m[:, :-1]; e[:, :-1] |= m[:, :-1] != m[:, 1:]
    return (e & m).sum()

def shape(m):
    """Máscara binaria -> área, aspecto bbox, solidez, compacidad (NaN si vacía)."""
    a = m.sum()
    if a < 20: return dict(area=a, aspect=np.nan, solidity=np.nan, compact=np.nan)
    ys, xs = np.nonzero(m)
    h, w = np.ptp(ys) + 1, np.ptp(xs) + 1
    try: hull = ConvexHull(np.c_[xs, ys]).volume
    except Exception: hull = np.nan
    p = perimeter(m)
    return dict(area=a, aspect=w / h, solidity=a / hull if hull else np.nan, compact=4 * np.pi * a / p ** 2 if p else np.nan)

def features(probs):
    """probs float [6,h,w] en [0,1] -> dict de características."""
    h, w = probs.shape[1:]; tot = h * w
    B = probs > 0.5; U = B.any(0); ua = U.sum()
    f = {"recorte_aspecto": w / h, "recorte_area_px": tot * 4,          # máscaras a media resolución
         "union_frac": ua / tot}
    su = shape(U); f["union_solidez"], f["union_compacidad"], f["union_aspecto"] = su["solidity"], su["compact"], su["aspect"]
    for i, r in enumerate(REGIONS):
        s = shape(B[i]); p = probs[i]
        f[f"{r}_frac_recorte"] = s["area"] / tot
        f[f"{r}_frac_union"] = s["area"] / ua if ua else np.nan
        f[f"{r}_aspecto"], f[f"{r}_solidez"], f[f"{r}_compacidad"] = s["aspect"], s["solidity"], s["compact"]
        f[f"{r}_confianza"] = p[B[i]].mean() if B[i].any() else np.nan
        band = (p > 0.1) & (p < 0.9)                                     # zona de transición del borde
        f[f"{r}_borde_difuso"] = band.sum() / max(B[i].sum(), 1)
    aL, aR = B[4].sum(), B[5].sum()
    f["muslos_simetria"] = min(aL, aR) / max(aL, aR) if max(aL, aR) else np.nan
    f["muslos_vs_grupa"] = (aL + aR) / B[2].sum() if B[2].any() else np.nan
    f["cadera_vs_grupa"] = B[1].sum() / B[2].sum() if B[2].any() else np.nan
    for a, b in PAIRS:
        i, j = REGIONS.index(a), REGIONS.index(b); inter = (B[i] & B[j]).sum()
        f[f"traslape_{a}_{b}"] = inter / max(min(B[i].sum(), B[j].sum()), 1)
    return f

def describe_name(c):
    if c.startswith("traslape_"):
        a, b = [r for r in REGIONS if r in c]
        return f"traslape {NOMBRES[a]} · {NOMBRES[b]}"
    for r in REGIONS: c = c.replace(r + "_", NOMBRES[r] + " · ")
    return c.replace("_", " ")

# ---------- Estadística ----------
def cluster_boot_spearman(x, y, g, n=2000, seed=0):
    rng = np.random.RandomState(seed); ug = np.unique(g); idx = {k: np.nonzero(g == k)[0] for k in ug}; out = []
    for _ in range(n):
        s = np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])
        ok = ~np.isnan(x[s])
        if ok.sum() > 10: out.append(spearmanr(x[s][ok], y[s][ok])[0])
    return np.nanpercentile(out, [2.5, 97.5])

def perm_spearman(x, y, n=10000, seed=0):
    ok = ~np.isnan(x); x, y = rankdata(x[ok]), rankdata(y[ok]); r = np.corrcoef(x, y)[0, 1]
    rng = np.random.RandomState(seed); null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(n)])
    return r, (np.sum(np.abs(null) >= abs(r)) + 1) / (n + 1)

def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty_like(q); out[o] = np.clip(q, 0, 1); return out

def icc1(x, g):
    ok = ~np.isnan(x); x, g = x[ok], g[ok]; ug = np.unique(g); k = len(x) / len(ug)
    gm = np.array([x[g == u].mean() for u in ug]); n_u = np.array([(g == u).sum() for u in ug])
    msb = np.sum(n_u * (gm - x.mean()) ** 2) / (len(ug) - 1)
    msw = sum(((x[g == u] - x[g == u].mean()) ** 2).sum() for u in ug) / (len(x) - len(ug))
    return (msb - msw) / (msb + (k - 1) * msw)

def loco_ridge(X, y, cows):
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X); p = np.zeros(len(y)); ug = np.unique(cows)
    for tr, te in GroupKFold(len(ug)).split(X, y, cows):
        gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": [0.1, 1, 10, 100, 1e3, 1e4]},
                          cv=GroupKFold(4), scoring="neg_mean_absolute_error")
        gs.fit(X[tr], y[tr], groups=cows[tr]); p[te] = gs.predict(X[te])
    return p

# ---------- Figura ----------
def plot(res, df, agg, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = res.sort_values("rho_foto"); n = len(r)
    fig = plt.figure(figsize=(15, max(9, n * 0.2)))
    ax = fig.add_axes([0.2, 0.05, 0.35, 0.9])
    yy = np.arange(n)
    ax.hlines(yy, r["ic95_lo"], r["ic95_hi"], color="#9aa7b4", lw=1.5, label="IC95 por foto (bootstrap de vacas)")
    ax.plot(r["rho_foto"], yy, "o", color="#2c5d8a", ms=4, label="ρ por foto")
    ax.plot(r["rho_animal"], yy, "D", mfc="none", color="#c0392b", ms=4, label="ρ por animal-estado")
    ax.axvline(0, color="k", lw=0.8); ax.set_yticks(yy); ax.set_yticklabels(r["descripcion"], fontsize=7)
    ax.set_xlim(-0.7, 0.7); ax.set_xlabel("Spearman ρ con BCS"); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Correlación característica de máscara – BCS", fontsize=10)
    top = res.head(4)["caracteristica"].tolist()
    for k, c in enumerate(top):
        a = fig.add_axes([0.62 + (k % 2) * 0.19, 0.55 - (k // 2) * 0.45, 0.15, 0.36])
        a.scatter(df["bcs"] + np.random.RandomState(k).uniform(-0.06, 0.06, len(df)), df[c], s=6, alpha=0.25, color="#2c5d8a")
        a.scatter(agg["bcs"], agg[c], s=22, color="#c0392b", edgecolor="w", lw=0.5)
        row = res.set_index("caracteristica").loc[c]
        a.set_title(f"{describe_name(c)}\nρ animal={row.rho_animal:.2f} (p={row.p_perm_animal:.3f}, q={row.q_fdr:.2f})", fontsize=8)
        a.set_xlabel("BCS", fontsize=8); a.tick_params(labelsize=7)
    fig.text(0.62, 0.97, "Top 4 por |ρ| animal-estado (azul: fotos; rojo: promedio por animal-estado)", fontsize=9)
    fig.savefig(out / "correlaciones.png", dpi=130); plt.close(fig)

# ---------- Main ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="../../../Dataset/Dataset_Fr_BCS")
    ap.add_argument("--work_dir", default="work")
    ap.add_argument("--out", default="work/analysis")
    a = ap.parse_args(); work, out = Path(a.work_dir), Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in load_records(a.data_dir):
        m = np.load(work / "masks" / Path(r["name"]).with_suffix(".npz"))["masks"].astype(np.float32) / 255
        cam = Path(r["name"]).stem.split("_cam")[-1][0]
        rows.append({"foto": r["name"], "vaca": r["vaca"], "bcs": r["bcs"], "camara": f"cam{cam}", **features(m)})
    df = pd.DataFrame(rows); df.to_csv(out / "mask_features.csv", index=False)
    feats = [c for c in df.columns if c not in ("foto", "vaca", "bcs", "camara")]
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]   # p.ej. cadera∩grupa siempre 0
    feats = [c for c in feats if c not in const]
    y, cows = df["bcs"].values, df["vaca"].values
    estado = (df["vaca"] + "_" + df["bcs"].astype(str)).values           # 8758 tiene 3 BCS distintos
    agg = df.groupby(estado)[feats + ["bcs"]].mean()

    res = []
    for c in feats:
        x = df[c].values.astype(float); ok = ~np.isnan(x)
        rho = spearmanr(x[ok], y[ok])[0]; lo, hi = cluster_boot_spearman(x, y, cows)
        rv, pv = perm_spearman(agg[c].values.astype(float), agg["bcs"].values)
        res.append({"caracteristica": c, "descripcion": describe_name(c), "rho_foto": rho, "ic95_lo": lo, "ic95_hi": hi,
                    "rho_animal": rv, "p_perm_animal": pv, "icc": icc1(x, estado), "faltantes": int((~ok).sum())})
    res = pd.DataFrame(res); res["q_fdr"] = bh(res["p_perm_animal"])
    res = res.reindex(res["rho_animal"].abs().sort_values(ascending=False).index)
    res.round(4).to_csv(out / "correlaciones.csv", index=False)
    plot(res, df, agg, out)

    # Cámara: ¿confusor?
    cam = df.groupby("camara")["bcs"].agg(["count", "mean", "std"])

    # ¿Cuánto BCS explica la geometría de las máscaras sola?
    X = df[feats].values.astype(float); p = loco_ridge(X, y, cows)
    p0 = np.array([y[cows != c].mean() for c in cows])
    mae, mae0 = np.abs(p - y).mean(), np.abs(p0 - y).mean()
    r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    pd.DataFrame({"foto": df["foto"], "vaca": cows, "bcs_real": y, "bcs_pred": p.round(4)}).to_csv(out / "preds_loco_mascaras.csv", index=False)

    pd.set_option("display.width", 200, "display.max_colwidth", 40)
    if const: print(f"[aviso] características constantes descartadas: {const}")
    print(f"\n{len(df)} fotos, {len(np.unique(cows))} vacas, {len(agg)} animal-estado | {len(feats)} características")
    print("\nTop 15 por |rho| a nivel animal-estado:")
    print(res.head(15)[["descripcion", "rho_foto", "ic95_lo", "ic95_hi", "rho_animal", "p_perm_animal", "q_fdr", "icc"]].round(3).to_string(index=False))
    print(f"\nSignificativas con FDR q<0.05: {int((res.q_fdr < 0.05).sum())} | q<0.10: {int((res.q_fdr < 0.10).sum())}")
    print("\nBCS por cámara:\n", cam.round(3))
    print(f"\nRidge LOCO solo con máscaras: MAE={mae:.3f}  R2={r2:.2f}  (media del train: {mae0:.3f}; DINOv2 por región: 0.583)")
    print(f"-> {out}/correlaciones.csv, correlaciones.png, mask_features.csv, preds_loco_mascaras.csv")

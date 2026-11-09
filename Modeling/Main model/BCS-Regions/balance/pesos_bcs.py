# =============================================================
# Nombre del archivo: balance/pesos_bcs.py
#
# Descripción:
#   Experimento para el modelo base: ¿los pesos por BCS reducen el
#   "jalar al centro" del ridge? Usa <work>/features.npz (no necesita GPU).
#   Ridge ponderado: min sum_i w_i (y_i - x_i b - c)^2 + alpha |b|^2.
#   Pesos (calculados solo con el train de cada fold, media 1):
#     - por BCS: bins de 0.5 (los mismos estratos del 5x10); peso de cada
#       bin = (peso total del bin)^-p, p en {0, 0.5, 1}
#     - por vaca (opcional): cada vaca pesa lo mismo (1 / fotos de la vaca)
#   alpha se elige como en bcs_regions.py (GroupKFold(4) por vaca dentro
#   del train) pero con MAE ponderado con los mismos pesos.
#   Métricas: las de bcs_regions.py + MAE balanceado (promedio del MAE por
#   bin de BCS), pendiente pred~real (1 = sin jalar al centro) y sesgo en
#   flacas (<=2.75) y gordas (>=4.5).
#   Salidas: <work>/balance/{resumen.csv, preds_<config>_<esquema>.csv}
#
# Uso: docker compose run --rm --entrypoint python bcs balance/pesos_bcs.py
# =============================================================
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br

CONFIGS = {"base": (0, False), "bcs_p0.5": (0.5, False), "bcs_p1": (1, False),
           "vaca": (0, True), "vaca_bcs_p0.5": (0.5, True), "vaca_bcs_p1": (1, True)}

def weights(y, cows, p, por_vaca):
    w = np.ones(len(y))
    if por_vaca:
        _, inv, cnt = np.unique(cows, return_inverse=True, return_counts=True); w = w / cnt[inv]
    if p:
        b = np.round(y * 2)
        for v in np.unique(b): m = b == v; w[m] *= w[m].sum() ** -p
    return w / w.mean()

def wridge(A, B, y, w, alphas):
    """Ridge ponderado en forma dual (n << p). Con w = 1 es igual a br.ridge_dual."""
    s = np.sqrt(w); mu = (w[:, None] * A).sum(0) / w.sum(); ym = (w * y).sum() / w.sum()
    Aw = (A - mu) * s[:, None]; ev, V = np.linalg.eigh(Aw @ Aw.T); Vy = V.T @ ((y - ym) * s); out = {}
    for al in alphas:
        coef = Aw.T @ (V @ (Vy / (ev + al))); out[al] = (B - mu) @ coef + ym
    return out

def fit_predict(Xtr, ytr, gtr, Xte, cfg):
    w = weights(ytr, gtr, *cfg); err = {al: [] for al in br.ALPHAS}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        A, B, _, _ = br.standardize(Xtr[i], Xtr[v])
        for al, p in wridge(A, B, ytr[i], w[i], br.ALPHAS).items(): err[al].append(w[v] * np.abs(p - ytr[v]))
    al = min(br.ALPHAS, key=lambda a_: np.mean(np.concatenate(err[a_])))
    A, B, _, _ = br.standardize(Xtr, Xte)
    return wridge(A, B, ytr, w, [al])[al], al

def extra(p, y):
    b = np.round(y * 2); slope = np.polyfit(y, p, 1)[0]
    return {"mae_balanceado": np.mean([np.abs(p - y)[b == v].mean() for v in np.unique(b)]), "pendiente": slope,
            "sesgo_flacas": (p - y)[y <= 2.75].mean(), "sesgo_gordas": (p - y)[y >= 4.5].mean()}

def main(a):
    work = Path(a.work_dir); out = work / "balance"; out.mkdir(exist_ok=True)
    X, y, cows, names = br.load_features(work); uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    rows = []
    for name, cfg in CONFIGS.items():
        t0 = time.time()
        for sch, reps in schemes.items():
            P, maes, als = [], [], []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits: p[te], al = fit_predict(X[tr], y[tr], cows[tr], X[te], cfg); als.append(al)
                P.append(p); maes.append(np.abs(p - y).mean())
            P = np.mean(P, 0); m = {**br.metrics(P, y, cows), **extra(P, y)}
            rows.append({"config": name, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                         "mae_reps_sd": round(float(np.std(maes)), 4), "alpha_mediana": float(np.median(als))})
            with open(out / f"preds_{name}_{sch}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
                w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, P)])
        print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    with open(out / "resumen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'config':14s} {'esquema':8s} {'MAE':>6s} {'MAEbal':>6s} {'R2fot':>6s} {'R2vac':>6s} {'±0.5':>5s} {'pend':>5s} {'flacas':>7s} {'gordas':>7s}")
    for r in rows:
        print(f"{r['config']:14s} {r['esquema']:8s} {r['mae']:6.3f} {r['mae_balanceado']:6.3f} {r['r2_fotos']:6.3f} {r['r2_vacas']:6.3f} "
              f"{r['acc_05']:5.0%} {r['pendiente']:5.2f} {r['sesgo_flacas']:+7.2f} {r['sesgo_gordas']:+7.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

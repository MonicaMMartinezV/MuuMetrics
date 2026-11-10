# =============================================================
# Nombre del archivo: exp_calibracion.py
#
# Descripción:
#   Experimento: corregir la compresión hacia el promedio del ridge.
#   Usa los descriptores ya calculados (<work>/features.npz), no toca
#   el modelo final.
#
#   Variantes:
#     base         ridge actual (alpha de ALPHAS, elegido por GroupKFold(4))
#     alpha_bajo   misma búsqueda pero con alphas más chicos (menos encogimiento)
#     cal_lineal   base + recalibración y = a + b·pred
#     cal_isot     base + recalibración isotónica (monótona)
#     ab+lineal    alpha_bajo + recalibración lineal
#   La recalibración se ajusta SOLO con predicciones fuera de fold del
#   train (GroupKFold(4) por vaca dentro del train del fold externo).
#
# Uso: docker compose run --rm --entrypoint python bcs exp_calibracion.py
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

ALPHAS_BAJO = (0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e5)

def fit_predict(Xtr, ytr, gtr, Xte, alphas):
    br.ALPHAS = alphas
    return br.fit_predict(Xtr, ytr, gtr, Xte)

def oof(X, y, g, alphas):
    """Predicciones fuera de fold dentro del train (para ajustar la recalibración)."""
    p = np.zeros(len(y))
    for i, v in GroupKFold(4).split(X, y, g): p[v], _ = fit_predict(X[i], y[i], g[i], X[v], alphas)
    return p

def run_fold(X, y, g, tr, te):
    out, alpha = {}, {}
    for name, al in [("base", br.ALPHAS_ORIG), ("alpha_bajo", ALPHAS_BAJO)]:
        out[name], alpha[name] = fit_predict(X[tr], y[tr], g[tr], X[te], al)
        po = oof(X[tr], y[tr], g[tr], al)
        b, a = np.polyfit(po, y[tr], 1)
        lin = np.clip(a + b * out[name], 1.0, 5.0)
        if name == "base":
            out["cal_lineal"] = lin
            out["cal_isot"] = IsotonicRegression(out_of_bounds="clip").fit(po, y[tr]).predict(out[name])
            alpha["pendiente"] = b
        else:
            out["ab+lineal"] = lin
    return out, alpha

def bias_by_range(p, y):
    return {f"sesgo_{lo}-{hi}": round(float((p - y)[(y >= lo) & (y < hi)].mean()), 3)
            for lo, hi in [(2, 3), (3, 3.75), (3.75, 4.5), (4.5, 5.1)]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); work = Path(a.work_dir)
    br.ALPHAS_ORIG = br.ALPHAS
    X, y, cows, names = br.load_features(work); uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    rows, loco_preds = [], None
    for sch, reps in schemes.items():
        P, info = {}, {"alpha_base": [], "alpha_bajo": [], "pendiente": []}
        for splits in reps:
            pr = {}
            for tr, te in splits:
                o, al = run_fold(X, y, cows, tr, te)
                for k, v in o.items(): pr.setdefault(k, np.zeros(len(y)))[te] = v
                info["alpha_base"].append(al["base"]); info["alpha_bajo"].append(al["alpha_bajo"]); info["pendiente"].append(al["pendiente"])
            for k, v in pr.items(): P.setdefault(k, []).append(v)
        P = {k: np.mean(v, 0) for k, v in P.items()}
        if sch == "loco": loco_preds = P
        print(f"\n[{sch}] alpha base (mediana)={np.median(info['alpha_base']):g}  alpha bajo (mediana)={np.median(info['alpha_bajo']):g}  "
              f"pendiente recal. (mediana)={np.median(info['pendiente']):.2f}")
        for k, p in P.items():
            m = br.metrics(p, y, cows); bb = bias_by_range(p, y)
            rows.append({"esquema": sch, "variante": k, **{kk: round(float(v), 4) for kk, v in m.items()}, **bb})
            print(f"  {k:11s} MAE={m['mae']:.3f}  RMSE={m['rmse']:.3f}  R2 fotos={m['r2_fotos']:.2f}  R2 vacas={m['r2_vacas']:.2f}  "
                  f"±0.5={m['acc_05']:.0%}  MAE vaca={m['mae_vaca']:.3f} [{m['mae_vaca_ic95_lo']:.3f},{m['mae_vaca_ic95_hi']:.3f}]  "
                  f"sesgo flacas(2-3)={bb['sesgo_2-3']:+.2f} gordas(4.5-5)={bb['sesgo_4.5-5.1']:+.2f}")
    out = work / "exp_calibracion"; out.mkdir(exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco_preds))
        for i in range(len(y)): w.writerow([names[i], cows[i], y[i]] + [round(float(loco_preds[k][i]), 4) for k in loco_preds])
    print(f"\n-> {out}")

if __name__ == "__main__":
    main()

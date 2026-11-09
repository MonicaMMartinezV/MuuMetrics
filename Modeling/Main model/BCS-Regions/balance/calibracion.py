# =============================================================
# Nombre del archivo: balance/calibracion.py
#
# Descripción:
#   Experimento para el modelo base: recalibración lineal del ridge para
#   quitar el "jalar al centro". Usa <work>/features.npz (no necesita GPU).
#   En cada fold externo, el GroupKFold(4) por vaca que ya elige alpha deja
#   predicciones fuera de fold del train; con ellas se ajusta una recta y se
#   aplica a la predicción del test:
#     base      sin recalibrar (igual a bcs_regions.py)
#     lineal    y ~ a + b*pred   (la corrección lineal de menor error cuadrático)
#     inversa   pred ~ c + s*y  ->  (pred - c) / s   (pendiente 1: quita todo el encogimiento)
#   Las predicciones se recortan a [1, 5].
#   Salidas: <work>/balance/calibracion_resumen.csv, calibracion_por_bcs.csv y preds_cal_*.csv
#
# Uso: docker compose run --rm --entrypoint python bcs balance/calibracion.py
# =============================================================
import argparse, csv, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from pesos_bcs import extra

MODES = ("base", "lineal", "inversa")

def fit_predict(Xtr, ytr, gtr, Xte):
    """-> {modo: pred_test}, alpha"""
    oof = {al: np.zeros(len(ytr)) for al in br.ALPHAS}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        A, B, _, _ = br.standardize(Xtr[i], Xtr[v])
        for al, (p, _) in br.ridge_dual(A, B, ytr[i], br.ALPHAS).items(): oof[al][v] = p
    al = min(br.ALPHAS, key=lambda a_: np.abs(oof[a_] - ytr).mean()); q = oof[al]
    A, B, _, _ = br.standardize(Xtr, Xte); p = br.ridge_dual(A, B, ytr, [al])[al][0]
    b, a = np.polyfit(q, ytr, 1); s, c = np.polyfit(ytr, q, 1)
    out = {"base": p, "lineal": a + b * p, "inversa": (p - c) / s}
    return {k: np.clip(v, 1, 5) for k, v in out.items()}, al

def main(a):
    work = Path(a.work_dir); out = work / "balance"; out.mkdir(exist_ok=True)
    X, y, cows, names = br.load_features(work); uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    rows, bins = [], []
    for sch, reps in schemes.items():
        P = {m: [] for m in MODES}; maes = {m: [] for m in MODES}
        for splits in reps:
            p = {m: np.zeros(len(y)) for m in MODES}
            for tr, te in splits:
                r, _ = fit_predict(X[tr], y[tr], cows[tr], X[te])
                for m in MODES: p[m][te] = r[m]
            for m in MODES: P[m].append(p[m]); maes[m].append(np.abs(p[m] - y).mean())
        for m in MODES:
            pm = np.mean(P[m], 0); met = {**br.metrics(pm, y, cows), **extra(pm, y)}
            rows.append({"calibracion": m, "esquema": sch, **{k: round(float(v), 4) for k, v in met.items()},
                         "mae_reps_sd": round(float(np.std(maes[m])), 4)})
            for v in np.unique(y):
                k = y == v; bins.append({"calibracion": m, "esquema": sch, "bcs": v, "vacas": len(np.unique(cows[k])), "fotos": int(k.sum()),
                                         "pred_media": round(float(pm[k].mean()), 3), "mae": round(float(np.abs(pm - y)[k].mean()), 3)})
            with open(out / f"preds_cal_{m}_{sch}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
                w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, pm)])
    for fn, data in (("calibracion_resumen.csv", rows), ("calibracion_por_bcs.csv", bins)):
        with open(out / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    print(f"{'calibracion':11s} {'esquema':8s} {'MAE':>6s} {'MAEbal':>6s} {'R2fot':>6s} {'R2vac':>6s} {'±0.5':>5s} {'pend':>5s} {'flacas':>7s} {'gordas':>7s} {'MAEvaca':>7s}")
    for r in rows:
        print(f"{r['calibracion']:11s} {r['esquema']:8s} {r['mae']:6.3f} {r['mae_balanceado']:6.3f} {r['r2_fotos']:6.3f} {r['r2_vacas']:6.3f} "
              f"{r['acc_05']:5.0%} {r['pendiente']:5.2f} {r['sesgo_flacas']:+7.2f} {r['sesgo_gordas']:+7.2f} {r['mae_vaca']:7.3f}")
    print("\nBCS   vacas  pred base / lineal / inversa (5x10)")
    for v in np.unique(y):
        d = {b["calibracion"]: b for b in bins if b["esquema"] == "rep5x10" and b["bcs"] == v}
        print(f"{v:.2f}  {d['base']['vacas']:3d}    {d['base']['pred_media']:.2f} / {d['lineal']['pred_media']:.2f} / {d['inversa']['pred_media']:.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

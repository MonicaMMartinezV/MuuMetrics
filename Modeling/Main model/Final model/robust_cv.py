# =============================================================
# Nombre del archivo: robust_cv.py
#
# Descripción:
#   Validación robusta de los modelos congelados + ridge (features
#   cacheadas por backbones_frozen.py / backbones_sweep.py):
#     rep5x10 : 5 folds agrupados por vaca y estratificados por BCS,
#               repetidos con 10 semillas.
#     loco    : dejar una vaca fuera (37 folds).
#   En ambos, alpha de ridge se elige con GroupKFold(4) por vaca dentro
#   del train de cada fold. Métricas con todas las predicciones juntas
#   (MAE, RMSE, R² por foto y por vaca, ±0.5) e IC 95 % por bootstrap
#   de vacas. Además, diferencia pareada de MAE contra el mejor modelo.
#
# Salidas (<out_dir>/robust): summary.csv, reps.csv, paired.csv,
#   preds_<esquema>_<exp>.csv
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from filmModel import load_records

ALPHAS = (1, 10, 100, 1e3, 1e4, 1e5)
FEATURE_FILES = {"dinov2": "features_dinov2_378.npz"}

def ridge_fit_predict(Xtr, ytr, gtr, Xte):
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    best, best_err = None, np.inf
    for a in ALPHAS:
        err = [np.abs(Ridge(alpha=a).fit(Xtr[i], ytr[i]).predict(Xtr[v]) - ytr[v])
               for i, v in GroupKFold(4).split(Xtr, ytr, gtr)]
        e = float(np.mean(np.concatenate(err)))
        if e < best_err: best, best_err = a, e
    return Ridge(alpha=best).fit(Xtr, ytr).predict(Xte)

def oof(X, y, cows, splits):
    p = np.zeros(len(y))
    for tr, te in splits: p[te] = ridge_fit_predict(X[tr], y[tr], cows[tr], X[te])
    return p

def r2(p, y): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def metrics(p, y, cows):
    uc = np.unique(cows); cp = np.array([p[cows == c].mean() for c in uc]); cy = np.array([y[cows == c].mean() for c in uc])
    rd = np.clip(np.round(p / 0.25) * 0.25, 2.0, 5.0)
    return {"mae": np.abs(p - y).mean(), "rmse": np.sqrt(((p - y) ** 2).mean()), "r2_fotos": r2(p, y),
            "r2_vacas": r2(cp, cy), "acc_05": np.mean(np.abs(rd - y) <= 0.5 + 1e-9)}

def cow_bootstrap(err_by_cow, n=5000, seed=0):
    """IC 95 % del promedio de una cantidad por vaca (cada vaca pesa igual)."""
    rng = np.random.RandomState(seed); k = len(err_by_cow)
    bs = [err_by_cow[rng.randint(0, k, k)].mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])

def main(a):
    t0 = time.time(); src = Path(a.out_dir); out = src / "robust"; out.mkdir(exist_ok=True)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    names = [Path(r["path"]).name for r in records]; uc = np.unique(cows)
    strata = np.round(y * 2).astype(int)                              # BCS en pasos de 0.5 para estratificar (clases)
    schemes = {f"rep5x10_s{s}": list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows))
               for s in range(10)}
    schemes["loco"] = list(GroupKFold(len(uc)).split(y, y, cows))

    exps = {}
    for spec in a.experiments.split(","):
        bb, pool = spec.split(":")
        f = np.load(src / FEATURE_FILES.get(bb, f"features_{bb}.npz"))
        exps[f"{bb}_{pool}_ridge"] = f["glob"] if pool == "global" else np.hstack([f["reg"], f["absent"]])

    rep_rows, preds = [], {}
    for exp, X in exps.items():
        for sch, splits in schemes.items():
            p = oof(X, y, cows, splits); preds[(exp, sch)] = p
            rep_rows.append({"experimento": exp, "esquema": sch, **{k: round(float(v), 4) for k, v in metrics(p, y, cows).items()}})
        print(f"{exp}: listo ({time.time()-t0:.0f}s)", flush=True)

    summary = []
    for exp in exps:
        for name, keys in (("rep5x10", [s for s in schemes if s.startswith("rep")]), ("loco", ["loco"])):
            P = np.mean([preds[(exp, s)] for s in keys], 0)             # predicción promedio de las repeticiones
            m = metrics(P, y, cows)
            reps = [r for r in rep_rows if r["experimento"] == exp and r["esquema"] in keys]
            lo, hi = cow_bootstrap(np.array([np.abs(P - y)[cows == c].mean() for c in uc]))
            summary.append({"experimento": exp, "esquema": name, **{k: round(float(v), 4) for k, v in m.items()},
                            "mae_reps_mean": round(float(np.mean([r["mae"] for r in reps])), 4),
                            "mae_reps_sd": round(float(np.std([r["mae"] for r in reps])), 4),
                            "mae_vaca": round(float(np.mean([np.abs(P - y)[cows == c].mean() for c in uc])), 4),
                            "mae_vaca_lo": round(float(lo), 4), "mae_vaca_hi": round(float(hi), 4)})
            with open(out / f"preds_{name}_{exp}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
                w.writerows([[n_, c, yy, round(float(pp), 4)] for n_, c, yy, pp in zip(names, cows, y, P)])

    paired = []
    for name in ("rep5x10", "loco"):
        rows = sorted([s for s in summary if s["esquema"] == name], key=lambda s: s["mae"])
        best = rows[0]["experimento"]
        keys = [s for s in schemes if s.startswith("rep")] if name == "rep5x10" else ["loco"]
        Pb = np.mean([preds[(best, s)] for s in keys], 0)
        for r in rows[1:]:
            Po = np.mean([preds[(r["experimento"], s)] for s in keys], 0)
            d = np.array([(np.abs(Pb - y) - np.abs(Po - y))[cows == c].mean() for c in uc])
            lo, hi = cow_bootstrap(d)
            paired.append({"esquema": name, "mejor": best, "contra": r["experimento"], "dif_mae_vaca": round(float(d.mean()), 4),
                           "ic_lo": round(float(lo), 4), "ic_hi": round(float(hi), 4), "significativo": bool(hi < 0)})

    for fn, rows in (("reps.csv", rep_rows), ("summary.csv", summary), ("paired.csv", paired)):
        with open(out / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name in ("rep5x10", "loco"):
        print(f"\n== {name}")
        for s in sorted([s for s in summary if s["esquema"] == name], key=lambda s: s["mae"]):
            print(f"{s['experimento']:28s} MAE={s['mae']:.3f} (reps {s['mae_reps_mean']:.3f}±{s['mae_reps_sd']:.3f}) "
                  f"MAE vaca={s['mae_vaca']:.3f} [{s['mae_vaca_lo']:.3f},{s['mae_vaca_hi']:.3f}] R2 fotos={s['r2_fotos']:.2f} vacas={s['r2_vacas']:.2f} ±.5={s['acc_05']:.2f}")
    for p_ in paired: print(f"[{p_['esquema']}] {p_['mejor']} vs {p_['contra']}: {p_['dif_mae_vaca']:+.3f} [{p_['ic_lo']:+.3f},{p_['ic_hi']:+.3f}] {'*' if p_['significativo'] else ''}")
    print(f"ROBUST DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv"); ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    ap.add_argument("--experiments", default="dinov2L:regiones,dinov2L:global,dinov2:regiones,dinov2:global,segencB:regiones,"
                                             "dinov3B:regiones,convnextB:regiones,dinov2Lreg:regiones,dinov2Lreg:global,dinov2g:regiones,dinov2g:global")
    main(ap.parse_args())

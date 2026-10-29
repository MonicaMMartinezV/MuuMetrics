# =============================================================
# Nombre del archivo: del_spline_cv.py
#
# Descripción:
#   ¿Agregar el DEL mejora al modelo DINOv2-L por región + ridge?
#   Compara, con el mismo protocolo que robust_cv.py (LOCO y 5 folds
#   por vaca estratificados x 10 semillas):
#     base         : regiones + indicadores de ausencia
#     del_lineal   : base + DEL
#     del_spline   : base + spline cúbico del DEL (4 nodos)
#   El DEL se estandariza y se multiplica por un peso w; alpha y w se
#   eligen JUNTOS con GroupKFold(4) por vaca dentro del train de cada
#   fold (nada se elige viendo las vacas de test). El spline también se
#   ajusta solo con el train.
#   Ridge resuelto en forma dual (n << p) para que la búsqueda sea rápida;
#   da lo mismo que sklearn Ridge con intercepto sobre X centrada.
#
# Salidas (<out_dir>/robust_del): summary.csv, paired.csv, chosen.csv,
#   preds_<esquema>_<exp>.csv
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer

from filmModel import load_records
from robust_cv import metrics, cow_bootstrap

ALPHAS = (1, 10, 100, 1e3, 1e4, 1e5)
WEIGHTS = (1, 3, 10, 30)

def standardize(tr, te):
    mu, sd = tr.mean(0), tr.std(0); sd[sd == 0] = 1
    return (tr - mu) / sd, (te - mu) / sd

def del_feats(kind, dtr, dte):
    if kind == "del_lineal": return dtr[:, None], dte[:, None]
    st = SplineTransformer(n_knots=4, degree=3).fit(dtr[:, None])
    return st.transform(dtr[:, None]), st.transform(dte[:, None])

def dual_ridge(Ktr, Kte, ytr, alphas):
    """Predicciones de ridge (X ya centrada) para varios alpha a partir de los kernels lineales."""
    ev, V = np.linalg.eigh(Ktr); yc = ytr - ytr.mean(); Vy = V.T @ yc
    return {a: Kte @ (V @ (Vy / (ev + a))) + ytr.mean() for a in alphas}

def fit_predict(Xi_tr, Xi_te, D_tr, D_te, ytr, gtr, weights):
    """Elige (alpha, w) con GroupKFold(4) por vaca dentro del train; regresa predicción de test y (alpha, w)."""
    cands = [(a, w) for w in weights for a in ALPHAS]
    err = {c: [] for c in cands}
    for i, v in GroupKFold(4).split(Xi_tr, ytr, gtr):
        A, B = standardize(Xi_tr[i], Xi_tr[v])
        Ki, Kv = A @ A.T, B @ A.T
        if D_tr is not None:
            Da, Db = standardize(D_tr[i], D_tr[v]); Di, Dv = Da @ Da.T, Db @ Da.T
        for w in weights:
            Kw, Kwv = (Ki, Kv) if D_tr is None else (Ki + w ** 2 * Di, Kv + w ** 2 * Dv)
            for a, p in dual_ridge(Kw, Kwv, ytr[i], ALPHAS).items(): err[(a, w)].append(np.abs(p - ytr[v]))
    best = min(cands, key=lambda c: np.mean(np.concatenate(err[c])))
    A, B = standardize(Xi_tr, Xi_te); K, Kt = A @ A.T, B @ A.T
    if D_tr is not None:
        Da, Db = standardize(D_tr, D_te); K, Kt = K + best[1] ** 2 * Da @ Da.T, Kt + best[1] ** 2 * Db @ Da.T
    return dual_ridge(K, Kt, ytr, [best[0]])[best[0]], best

def main(a):
    t0 = time.time(); out = Path(a.out_dir) / "robust_del"; out.mkdir(parents=True, exist_ok=True)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    d = np.array([r["del_days"] for r in records], float); names = [Path(r["path"]).name for r in records]
    f = np.load(a.features); X = np.hstack([f["reg"], f["absent"]]); uc = np.unique(cows)
    strata = np.round(y * 2).astype(int)
    schemes = {f"rep5x10_s{s}": list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(10)}
    schemes["loco"] = list(GroupKFold(len(uc)).split(y, y, cows))

    preds, chosen = {}, []
    for exp in ("base", "del_lineal", "del_spline"):
        for sch, splits in schemes.items():
            p = np.zeros(len(y))
            for k, (tr, te) in enumerate(splits):
                Dtr, Dte = (None, None) if exp == "base" else del_feats(exp, d[tr], d[te])
                p[te], (al, w) = fit_predict(X[tr], X[te], Dtr, Dte, y[tr], cows[tr], (1,) if exp == "base" else WEIGHTS)
                chosen.append({"experimento": exp, "esquema": sch, "fold": k, "alpha": al, "peso_del": w if exp != "base" else ""})
            preds[(exp, sch)] = p
        print(f"{exp}: listo ({time.time()-t0:.0f}s)", flush=True)

    summary, paired = [], []
    for name, keys in (("rep5x10", [s for s in schemes if s.startswith("rep")]), ("loco", ["loco"])):
        P = {e: np.mean([preds[(e, s)] for s in keys], 0) for e in ("base", "del_lineal", "del_spline")}
        for e, pe in P.items():
            m = metrics(pe, y, cows); ec = np.array([np.abs(pe - y)[cows == c].mean() for c in uc]); lo, hi = cow_bootstrap(ec)
            reps = [np.abs(preds[(e, s)] - y).mean() for s in keys]
            summary.append({"experimento": e, "esquema": name, **{k: round(float(v), 4) for k, v in m.items()},
                            "mae_reps_sd": round(float(np.std(reps)), 4), "mae_vaca": round(float(ec.mean()), 4),
                            "mae_vaca_lo": round(float(lo), 4), "mae_vaca_hi": round(float(hi), 4)})
            with open(out / f"preds_{name}_{e}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "del", "bcs_real", "bcs_pred"])
                w.writerows([[n_, c, int(dd), yy, round(float(pp), 4)] for n_, c, dd, yy, pp in zip(names, cows, d, y, pe)])
        for e in ("del_lineal", "del_spline"):
            dif = np.array([(np.abs(P[e] - y) - np.abs(P["base"] - y))[cows == c].mean() for c in uc]); lo, hi = cow_bootstrap(dif)
            paired.append({"esquema": name, "modelo": e, "contra": "base", "dif_mae_vaca": round(float(dif.mean()), 4),
                           "ic_lo": round(float(lo), 4), "ic_hi": round(float(hi), 4), "significativo": bool(hi < 0 or lo > 0),
                           "vacas_mejoran": int((dif < 0).sum()), "vacas_empeoran": int((dif > 0).sum())})

    for fn, rows in (("summary.csv", summary), ("paired.csv", paired), ("chosen.csv", chosen)):
        with open(out / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for s in summary:
        print(f"[{s['esquema']:7s}] {s['experimento']:10s} MAE={s['mae']:.3f} (sd reps {s['mae_reps_sd']:.3f}) MAE vaca={s['mae_vaca']:.3f} "
              f"[{s['mae_vaca_lo']:.3f},{s['mae_vaca_hi']:.3f}] R2 fotos={s['r2_fotos']:.2f} vacas={s['r2_vacas']:.2f} ±.5={s['acc_05']:.2f}")
    for p_ in paired:
        print(f"[{p_['esquema']}] {p_['modelo']} - base: {p_['dif_mae_vaca']:+.3f} [{p_['ic_lo']:+.3f},{p_['ic_hi']:+.3f}] "
              f"{'*' if p_['significativo'] else ''} mejoran {p_['vacas_mejoran']}, empeoran {p_['vacas_empeoran']}")
    for e in ("del_lineal", "del_spline"):
        ws = [c["peso_del"] for c in chosen if c["experimento"] == e]
        print(f"{e}: peso elegido " + ", ".join(f"x{w}: {ws.count(w)}" for w in WEIGHTS))
    print(f"DEL DONE {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv"); ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--features", default="outputs_backbones_cv/features_dinov2L.npz")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    main(ap.parse_args())

# =============================================================
# Nombre del archivo: del_baseline.py
#
# Descripción:
#   Punto de partida del frente DEL: ¿agregar el DEL al modelo de
#   regiones mejora el BCS? Compara, con la misma validación que
#   `bcs_regions.py evaluate` (LOCO y 5 folds por vaca x N semillas):
#     base        regiones (features.npz)
#     del_lineal  base + DEL
#     del_spline  base + spline cúbico del DEL (4 nodos, ajustado en el train)
#   El DEL estandarizado se multiplica por un peso w; alpha y w se eligen
#   juntos con GroupKFold(4) por vaca dentro del train de cada fold.
#   DEL faltante -> mediana del train.
#   Resultado previo (rama Adri, 368 fotos): ninguna variante mejora.
#
# Uso (desde Modeling/Main model/BCS-Regions, tras `bcs all` y del/build_del.py):
#   docker compose run --rm --entrypoint python bcs del/del_baseline.py
# Salida: <work>/del_baseline.csv
# =============================================================
import argparse, csv, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bcs_regions import load_features, standardize, ridge_dual, metrics, ALPHAS
from del_data import load_del

WEIGHTS = (1, 3, 10, 30)

def del_cols(kind, dtr, dte):
    med = np.nanmedian(dtr); dtr, dte = np.where(np.isnan(dtr), med, dtr), np.where(np.isnan(dte), med, dte)
    if kind == "del_lineal": return dtr[:, None], dte[:, None]
    st = SplineTransformer(n_knots=4, degree=3).fit(dtr[:, None])
    return st.transform(dtr[:, None]), st.transform(dte[:, None])

def design(Xtr, Xte, Dtr, Dte, w):
    A, B, _, _ = standardize(Xtr, Xte)
    if Dtr is None: return A, B
    Da, Db, _, _ = standardize(Dtr, Dte)
    return np.hstack([A, w * Da]), np.hstack([B, w * Db])

def fit_predict(kind, Xtr, ytr, gtr, dtr, Xte, dte):
    weights = (1,) if kind == "base" else WEIGHTS
    err = {(a, w): [] for w in weights for a in ALPHAS}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        Di, Dv = (None, None) if kind == "base" else del_cols(kind, dtr[i], dtr[v])
        for w in weights:
            A, B = design(Xtr[i], Xtr[v], Di, Dv, w)
            for a, (p, _) in ridge_dual(A, B, ytr[i], ALPHAS).items(): err[(a, w)].append(np.abs(p - ytr[v]))
    a, w = min(err, key=lambda k: np.mean(np.concatenate(err[k])))
    Dtr, Dte = (None, None) if kind == "base" else del_cols(kind, dtr, dte)
    A, B = design(Xtr, Xte, Dtr, Dte, w)
    return ridge_dual(A, B, ytr, [a])[a][0]

def main(a):
    work = Path(a.work_dir); X, y, cows, names = load_features(work); d, _ = load_del(names, work); uc = np.unique(cows)
    print(f"{len(y)} fotos, {len(uc)} vacas, {int(np.isnan(d).sum())} sin DEL")
    strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(X, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, strata, cows)) for s in range(a.seeds)]}
    rows = []
    for sch, reps in schemes.items():
        P = {}
        for kind in ("base", "del_lineal", "del_spline"):
            ps = []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits: p[te] = fit_predict(kind, X[tr], y[tr], cows[tr], d[tr], X[te], d[te])
                ps.append(p)
            P[kind] = np.mean(ps, 0)
        for kind, p in P.items():
            m = metrics(p, y, cows); row = {"esquema": sch, "modelo": kind, **{k: round(float(v), 4) for k, v in m.items()}}
            if kind != "base":   # diferencia pareada por vaca contra base (positivo = peor), IC por bootstrap de vacas
                dif = np.array([(np.abs(p - y) - np.abs(P["base"] - y))[cows == c].mean() for c in uc])
                rng = np.random.RandomState(0); bs = [dif[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]
                lo, hi = np.percentile(bs, [2.5, 97.5])
                row.update(dif_vs_base=round(float(dif.mean()), 4), dif_ic95_lo=round(float(lo), 4), dif_ic95_hi=round(float(hi), 4))
            rows.append(row)
            extra = f"  vs base {row['dif_vs_base']:+.3f} [{row['dif_ic95_lo']:+.3f}, {row['dif_ic95_hi']:+.3f}]" if kind != "base" else ""
            print(f"[{sch:7s}] {kind:10s} MAE={row['mae']:.3f}  R2 vacas={row['r2_vacas']:.2f}{extra}")
    keys = list(rows[-1])
    with open(work / "del_baseline.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"-> {work / 'del_baseline.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work"); ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

# =============================================================
# Experimento [22]: curva de aprendizaje. ¿Cuánto mejora el modelo actual
# (224 px / bloque 17 + ridge) con más vacas?
#   8 folds por vaca (8 vacas de prueba, 56 de train) x SEEDS semillas;
#   del train se usa un subconjunto aleatorio de n vacas, n en NS.
#   Las vacas de prueba son las mismas para todos los n (comparación pareada).
# Ajuste: MAE(n) = a + b / sqrt(n) para extrapolar a 100-200 vacas
# (orientativo: la extrapolación supone que las vacas nuevas se parecen).
# Uso: docker compose run --rm --entrypoint python bcs exp_curva.py
# =============================================================
import csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
import bcs_regions as br

NS, SEEDS = [8, 16, 24, 32, 40, 48, 56], 10
X, y, cows, _ = br.load_features(Path("/work")); uc = np.unique(cows); out = Path("/work/exp_curva"); out.mkdir(exist_ok=True)
rows = []
for s in range(SEEDS):
    rng = np.random.default_rng(s); perm = rng.permutation(uc); fold = {c: i % 8 for i, c in enumerate(perm)}
    f = np.array([fold[c] for c in cows])
    for k in range(8):
        te = f == k; trc = np.array([c for c in perm if fold[c] != k])
        for n in NS:
            sub = rng.choice(trc, n, replace=False); tr = np.isin(cows, sub)
            p, al = br.fit_predict(X[tr], y[tr], cows[tr], X[te])
            e = p - y[te]; ct = cows[te]
            rows.append({"seed": s, "fold": k, "n_vacas": n, "fotos_train": int(tr.sum()), "alpha": al,
                         "mae": np.abs(e).mean(), "mae_vaca": np.mean([np.abs(e[ct == c]).mean() for c in np.unique(ct)]),
                         "sse": (e ** 2).sum(), "sst": ((y[te] - y.mean()) ** 2).sum(),
                         "sesgo_flacas": e[y[te] < 3].mean() if (y[te] < 3).any() else np.nan})
    print(f"semilla {s} lista", flush=True)
with open(out / "curva.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"\n{'vacas':>5s} {'MAE foto':>9s} {'MAE vaca':>9s} {'R2':>5s} {'sesgo flacas':>12s}")
M = []
for n in NS:
    r = [q for q in rows if q["n_vacas"] == n]
    by_seed = [np.mean([q["mae"] for q in r if q["seed"] == s]) for s in range(SEEDS)]
    r2 = 1 - sum(q["sse"] for q in r) / sum(q["sst"] for q in r)
    M.append(np.mean([q["mae"] for q in r]))
    print(f"{n:5d} {M[-1]:9.3f} {np.mean([q['mae_vaca'] for q in r]):9.3f} {r2:5.2f} {np.nanmean([q['sesgo_flacas'] for q in r]):+12.2f}  (sd entre semillas {np.std(by_seed):.3f})")
A = np.c_[np.ones(len(NS)), 1 / np.sqrt(NS)]; a, b = np.linalg.lstsq(A, np.array(M), rcond=None)[0]
print(f"\najuste MAE = {a:.3f} + {b:.3f}/sqrt(n) -> " + ", ".join(f"{n} vacas: {a + b / np.sqrt(n):.3f}" for n in [64, 100, 150, 200, 400]))

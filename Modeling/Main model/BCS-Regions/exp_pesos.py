# =============================================================
# Nombre del archivo: exp_pesos.py
#
# Descripción:
#   Experimento: dar el mismo peso a cada vaca. Hay vacas con 1 foto y
#   otras con 48, así que en el ridge sin pesos unas pocas vacas dominan.
#
#   Ridge ponderado (forma dual, igual que bcs_regions.ridge_dual):
#     min  sum_i w_i (y_i - b0 - x_i·b)^2 + alpha |b|^2
#   con pesos normalizados a promedio 1. El alpha se elige con
#   GroupKFold(4) por vaca minimizando el MAE ponderado con los mismos
#   pesos (para 1/n = MAE por vaca).
#
#   Pesos:  ninguno | 1/sqrt(n_vaca) | 1/n_vaca
#   Descriptores: base (work/features.npz) y capa18+espejo
#   (work/exp_descriptores/feats.npz, ver exp_descriptores.py).
#
# Uso: docker compose run --rm --entrypoint python bcs exp_pesos.py
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br
import exp_descriptores as ed

ALPHAS = br.ALPHAS

def cow_weights(cows, power):
    _, inv, cnt = np.unique(cows, return_inverse=True, return_counts=True)
    w = cnt[inv].astype(float) ** -power
    return w / w.mean()

def wridge(Xtr, ytr, w, Xte, alphas):
    """-> {alpha: pred_te}. Estandariza con media/sd ponderadas."""
    mu = np.average(Xtr, 0, w); sd = np.sqrt(np.average((Xtr - mu) ** 2, 0, w)); sd[sd == 0] = 1
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    ym = np.average(ytr, weights=w); s = np.sqrt(w)
    As = A * s[:, None]; ev, V = np.linalg.eigh(As @ As.T); Vy = V.T @ ((ytr - ym) * s)
    return {al: B @ (As.T @ (V @ (Vy / (ev + al)))) + ym for al in alphas}

def fit_predict(Xtr, ytr, gtr, Xte, power):
    w = cow_weights(gtr, power); err = {al: [0.0, 0.0] for al in ALPHAS}
    for i, v in GroupKFold(4).split(Xtr, ytr, gtr):
        wv = cow_weights(gtr[v], power)
        for al, p in wridge(Xtr[i], ytr[i], w[i] / w[i].mean(), Xtr[v], ALPHAS).items():
            err[al][0] += (wv * np.abs(p - ytr[v])).sum(); err[al][1] += wv.sum()
    al = min(ALPHAS, key=lambda a: err[a][0] / err[a][1])
    return wridge(Xtr, ytr, w, Xte, [al])[al], al

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); work = Path(a.work_dir)
    Xb, y, cows, names = br.load_features(work)
    z = dict(np.load(work / "exp_descriptores" / "feats.npz"))
    assert (z["name"] == names).all()
    feats = {"base": Xb, "capa18+espejo": ed.build(z, "squash", True, [1], False)}
    uc = np.unique(cows); strata = np.round(y * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(uc)).split(y, y, cows))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(a.seeds)]}
    rows, loco = [], {}
    for fname, X in feats.items():
        for wname, power in [("sin peso", 0.0), ("1/sqrt(n)", 0.5), ("1/n", 1.0)]:
            res, alphas = {}, []
            for sch, reps in schemes.items():
                P = []
                for splits in reps:
                    p = np.zeros(len(y))
                    for tr, te in splits: p[te], al = fit_predict(X[tr], y[tr], cows[tr], X[te], power); alphas.append(al)
                    P.append(p)
                res[sch] = np.mean(P, 0)
            name = f"{fname} | {wname}"; loco[name] = res["loco"]; line = []
            for sch, p in res.items():
                m = br.metrics(p, y, cows); e = p - y
                rows.append({"descriptores": fname, "pesos": wname, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                             "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3)})
                line.append(f"{sch}: MAE foto={m['mae']:.3f} MAE vaca={m['mae_vaca']:.3f} R2 vacas={m['r2_vacas']:.2f}")
            print(f"{name:28s} alpha~{np.median(alphas):g} | " + " | ".join(line) +
                  f" | flacas={rows[-2]['sesgo_flacas']:+.2f} gordas={rows[-2]['sesgo_gordas']:+.2f}", flush=True)

    rng = np.random.RandomState(0); per_cow = lambda p: np.array([np.abs(p - y)[cows == u].mean() for u in uc])
    ref = per_cow(loco["base | sin peso"])
    print("\nDiferencia de MAE por vaca vs 'base | sin peso' (LOCO, IC95 bootstrap por vaca):")
    for n, p in loco.items():
        if n == "base | sin peso": continue
        d = per_cow(p) - ref; bs = [d[rng.randint(0, len(uc), len(uc))].mean() for _ in range(5000)]
        print(f"  {n:28s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  mejoran {np.sum(d < 0)}/{len(uc)}")

    out = work / "exp_pesos"; out.mkdir(exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco))
        for i in range(len(y)): w.writerow([names[i], cows[i], y[i]] + [round(float(loco[k][i]), 4) for k in loco])
    print(f"-> {out}")

if __name__ == "__main__":
    main()

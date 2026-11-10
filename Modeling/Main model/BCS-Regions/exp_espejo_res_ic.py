# =============================================================
# Nombre del archivo: exp_espejo_res_ic.py
#
# Descripción:
#   Experimento [26], complemento: IC95 de las diferencias de MAE y R²
#   (por foto y por vaca) de cada variante contra la base 384 px sin
#   espejo, con bootstrap POR VACA (se remuestrean vacas con todas sus
#   fotos, porque las fotos de una vaca no son independientes).
#   Lee <work>/exp_espejo_res/preds_loco.csv -> ic_bootstrap.csv
#
# Uso: docker compose run --rm --entrypoint python bcs exp_espejo_res_ic.py
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np

def r2(p, y): return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def stats(p, y, cows, uc):
    cp = np.array([p[cows == c].mean() for c in uc]); cy = np.array([y[cows == c].mean() for c in uc])
    mv = np.array([np.abs(p - y)[cows == c].mean() for c in uc])
    return np.array([np.abs(p - y).mean(), r2(p, y), mv.mean(), r2(cp, cy)])

NAMES = ["MAE foto", "R2 foto", "MAE vaca", "R2 vaca"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--work_dir", default="/work"); ap.add_argument("--n", type=int, default=5000)
    a = ap.parse_args(); out = Path(a.work_dir) / "exp_espejo_res"
    with open(out / "preds_loco.csv") as fh: rows = list(csv.DictReader(fh))
    cows = np.array([r["vaca"] for r in rows]); y = np.array([float(r["bcs_real"]) for r in rows])
    variants = [k for k in rows[0] if k not in ("foto", "vaca", "bcs_real")]; base = "384px sin espejo"
    P = {k: np.array([float(r[k]) for r in rows]) for k in variants}
    uc = np.unique(cows); idx = {c: np.nonzero(cows == c)[0] for c in uc}
    rng = np.random.RandomState(0); samples = [rng.randint(0, len(uc), len(uc)) for _ in range(a.n)]
    res = []
    for k in variants:
        if k == base: continue
        obs = stats(P[k], y, cows, uc) - stats(P[base], y, cows, uc); bs = []
        for s in samples:
            ii = np.concatenate([idx[uc[j]] for j in s]); cc = np.concatenate([[f"{j}_{n}"] * len(idx[uc[j]]) for n, j in enumerate(s)])
            u2 = np.unique(cc); bs.append(stats(P[k][ii], y[ii], cc, u2) - stats(P[base][ii], y[ii], cc, u2))
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        print(f"{k:18s} " + " | ".join(f"Δ{n} {o:+.3f} [{l:+.3f}, {h:+.3f}]" for n, o, l, h in zip(NAMES, obs, lo, hi)), flush=True)
        for n, o, l, h in zip(NAMES, obs, lo, hi): res.append({"variante": k, "metrica": n, "dif": round(o, 4), "ic_lo": round(l, 4), "ic_hi": round(h, 4)})
    with open(out / "ic_bootstrap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(f"-> {out / 'ic_bootstrap.csv'}")

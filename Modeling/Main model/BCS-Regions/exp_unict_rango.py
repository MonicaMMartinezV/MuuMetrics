# =============================================================
# Nombre del archivo: exp_unict_rango.py
# Descripción: [11] como exp_unict.py, pero solo se suman al train las vacas
#   UNICT cuyo BCS cae en el hueco del robot. Evaluación siempre en el robot.
# Uso: python exp_unict_rango.py --work_dir /work
# =============================================================
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
import bcs_regions as br

def per_cow(p, y, c, uc): return np.array([np.abs(p - y)[c == u].mean() for u in uc])

ap = argparse.ArgumentParser(); ap.add_argument("--work_dir", default="/work"); ap.add_argument("--seeds", type=int, default=10)
a = ap.parse_args(); work = Path(a.work_dir)
Xr, yr, cr, nr = br.load_features(work / "robot_seg2"); Xu, yu, cu, nu = br.load_features(work / "unict")
RANGOS = {"[2.75, 3.75]": (2.75, 3.75), "[3, 3.5]": (3.0, 3.5), "[2.75, 3.5]": (2.75, 3.5)}
ucr = np.unique(cr); strata = np.round(yr * 2).astype(int)
schemes = {"loco": [list(GroupKFold(len(ucr)).split(yr, yr, cr))],
           "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yr, strata, cr)) for s in range(a.seeds)]}
variants = {"seg2": None}
for n, (lo, hi) in RANGOS.items():
    idx = np.where((yu >= lo - 1e-9) & (yu <= hi + 1e-9))[0]; variants[f"+UNICT {n}"] = idx
    print(f"UNICT {n}: {len(np.unique(cu[idx]))} vacas, {len(idx)} fotos, BCS {sorted(set(yu[idx].tolist()))}")
gap = (yr >= 2.75) & (yr <= 3.75); print(f"robot en [2.75, 3.75]: {len(np.unique(cr[gap]))} vacas, {gap.sum()} fotos\n")
loco = {}
for name, idx in variants.items():
    out = []
    for sch, reps in schemes.items():
        P = []
        for splits in reps:
            p = np.zeros(len(yr))
            for tr, te in splits:
                Xtr, ytr, gtr = Xr[tr], yr[tr], cr[tr]
                if idx is not None: Xtr = np.vstack([Xtr, Xu[idx]]); ytr = np.concatenate([ytr, yu[idx]]); gtr = np.concatenate([gtr, cu[idx]])
                p[te], _ = br.fit_predict(Xtr, ytr, gtr, Xr[te])
            P.append(p)
        p = np.mean(P, 0); m = br.metrics(p, yr, cr); e = p - yr
        if sch == "loco": loco[name] = p
        out.append(f"{sch} MAE={m['mae']:.3f} vaca={m['mae_vaca']:.3f} | hueco MAE={np.abs(e[gap]).mean():.3f} sesgo={e[gap].mean():+.2f} | flacas<2.75 sesgo={e[yr < 2.75].mean():+.2f}")
    print(f"{name:22s} " + "  ||  ".join(out), flush=True)
rng = np.random.RandomState(0); ref = per_cow(loco["seg2"], yr, cr, ucr); ug = np.unique(cr[gap]); refg = per_cow(loco["seg2"], yr, cr, ug)
print("\nDiferencia MAE por vaca vs seg2 (LOCO, IC95 bootstrap): todas las vacas | solo vacas del hueco")
for n, p in loco.items():
    if n == "seg2": continue
    d = per_cow(p, yr, cr, ucr) - ref; bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
    dg = per_cow(p, yr, cr, ug) - refg; bg = [dg[rng.randint(0, len(dg), len(dg))].mean() for _ in range(5000)]
    print(f"  {n:22s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] mejoran {np.sum(d < 0)}/{len(d)}"
          f"  |  {dg.mean():+.3f} [{np.percentile(bg, 2.5):+.3f}, {np.percentile(bg, 97.5):+.3f}] mejoran {np.sum(dg < 0)}/{len(dg)}")
    print("      vacas hueco:", ", ".join(f"{u}({yr[cr == u][0]:g}) {refg[k]:.2f}->{refg[k] + dg[k]:.2f}" for k, u in enumerate(ug)))

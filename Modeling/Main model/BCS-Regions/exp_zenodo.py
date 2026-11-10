# =============================================================
# Nombre del archivo: exp_zenodo.py
#
# Descripción:
#   Experimento: agregar al entrenamiento del ridge las fotos de Zenodo
#   (sala rotativa, 06.12.2024; recortes de prep_zenodo.py ya filtrados).
#   La evaluación es SIEMPRE sobre las vacas del robot que quedan fuera;
#   Zenodo solo se suma al train de cada fold.
#
#   Entradas:  <work>/features.npz         (robot, paso 'features')
#              <work>/zenodo/features.npz  (Zenodo, mismo paso con
#                                           --work_dir /work/zenodo)
#   Variantes:
#     base           solo robot (modelo actual)
#     +zen k=5       + 5 fotos por carpeta de Zenodo (repartidas en el tiempo)
#     +zen k=20      + todas las fotos de Zenodo
#     +zen k=5 +ind  + 5 por carpeta y un indicador de origen (robot/Zenodo)
#   Además, como referencia: modelo solo-robot prediciendo Zenodo y
#   modelo solo-Zenodo (5 folds por vaca) prediciendo Zenodo.
#
# Uso: docker compose run --rm --entrypoint python bcs exp_zenodo.py
# =============================================================
import argparse, csv, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

def subsample(names, k):
    """k fotos por carpeta de Zenodo (nombre: z<vaca>_<bcs>/<carpeta>_<hora>.jpg), espaciadas en el tiempo."""
    groups = {}
    for i, n in enumerate(names): groups.setdefault(n.split("/")[0] + "/" + n.split("/")[1].split("_")[0], []).append(i)
    idx = []
    for g in groups.values():
        g = sorted(g, key=lambda i: names[i]); idx += [g[j] for j in np.unique(np.linspace(0, len(g) - 1, min(k, len(g))).round().astype(int))]
    return np.array(sorted(idx))

def per_cow(p, y, cows, uc): return np.array([np.abs(p - y)[cows == u].mean() for u in uc])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); work = Path(a.work_dir); t0 = time.time()
    Xr, yr, cr, nr = br.load_features(work)
    Xz, yz, cz, nz = br.load_features(work / "zenodo")
    print(f"[datos] robot {len(yr)} fotos / {len(np.unique(cr))} vacas | Zenodo {len(yz)} fotos / {len(np.unique(cz))} vacas, BCS {yz.min()}-{yz.max()}")
    ucr = np.unique(cr); strata = np.round(yr * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(ucr)).split(yr, yr, cr))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yr, strata, cr)) for s in range(a.seeds)]}
    i5 = subsample(nz, 5)
    variants = {"base": None, "+zen k=5": (i5, False), "+zen k=20": (np.arange(len(yz)), False), "+zen k=5 +ind": (i5, True)}
    rows, loco = [], {}
    for name, extra in variants.items():
        res = {}
        for sch, reps in schemes.items():
            if name == "+zen k=20" and sch == "rep5x10": continue   # caro y redundante con k=5
            P = []
            for splits in reps:
                p = np.zeros(len(yr))
                for tr, te in splits:
                    Xtr, ytr, gtr, Xte = Xr[tr], yr[tr], cr[tr], Xr[te]
                    if extra is not None:
                        idx, ind = extra
                        Xtr = np.vstack([Xtr, Xz[idx]]); ytr = np.concatenate([ytr, yz[idx]]); gtr = np.concatenate([gtr, cz[idx]])
                        if ind:
                            src = np.r_[np.zeros(len(tr)), np.ones(len(idx))]
                            Xtr = np.c_[Xtr, src]; Xte = np.c_[Xte, np.zeros(len(te))]
                    p[te], _ = br.fit_predict(Xtr, ytr, gtr, Xte)
                P.append(p)
            res[sch] = np.mean(P, 0)
        loco[name] = res["loco"]; line = []
        for sch, p in res.items():
            m = br.metrics(p, yr, cr); e = p - yr
            rows.append({"variante": name, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_2-3": round(float(e[yr < 3].mean()), 3), "sesgo_3-3.75": round(float(e[(yr >= 3) & (yr < 3.75)].mean()), 3),
                         "sesgo_3.75-4.5": round(float(e[(yr >= 3.75) & (yr < 4.5)].mean()), 3), "sesgo_4.5-5": round(float(e[yr >= 4.5].mean()), 3)})
            line.append(f"{sch}: MAE={m['mae']:.3f} R2={m['r2_fotos']:.2f} ±0.5={m['acc_05']:.0%} MAE vaca={m['mae_vaca']:.3f}")
        r = [x for x in rows if x["variante"] == name and x["esquema"] == "loco"][0]
        print(f"{name:14s} | " + " | ".join(line) + f" | sesgo 2-3={r['sesgo_2-3']:+.2f} 3-3.75={r['sesgo_3-3.75']:+.2f} "
              f"3.75-4.5={r['sesgo_3.75-4.5']:+.2f} 4.5-5={r['sesgo_4.5-5']:+.2f} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.RandomState(0); ref = per_cow(loco["base"], yr, cr, ucr)
    print("\nDiferencia de MAE por vaca del robot vs base (LOCO, IC95 bootstrap por vaca):")
    for n, p in loco.items():
        if n == "base": continue
        d = per_cow(p, yr, cr, ucr) - ref; bs = [d[rng.randint(0, len(ucr), len(ucr))].mean() for _ in range(5000)]
        print(f"  {n:14s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  mejoran {np.sum(d < 0)}/{len(ucr)}")

    # Referencias sobre Zenodo
    pz, _ = br.fit_predict(Xr, yr, cr, Xz)
    print(f"\n[ref] solo robot -> Zenodo: MAE={np.abs(pz - yz).mean():.3f}  sesgo={np.mean(pz - yz):+.2f}  R2={br.r2(pz, yz):.2f}")
    pz2 = np.zeros(len(yz)); Xs, ys, cs = Xz[i5], yz[i5], cz[i5]; pz2 = np.zeros(len(ys))
    for tr, te in GroupKFold(5).split(Xs, ys, cs): pz2[te], _ = br.fit_predict(Xs[tr], ys[tr], cs[tr], Xs[te])
    mean_mae = np.abs(ys - ys.mean()).mean()
    print(f"[ref] solo Zenodo -> Zenodo (5 folds por vaca, k=5): MAE={np.abs(pz2 - ys).mean():.3f}  R2={br.r2(pz2, ys):.2f}  (predecir la media: {mean_mae:.3f})")

    out = work / "exp_zenodo"; out.mkdir(exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco))
        for i in range(len(yr)): w.writerow([nr[i], cr[i], yr[i]] + [round(float(loco[k][i]), 4) for k in loco])
    print(f"-> {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    main()

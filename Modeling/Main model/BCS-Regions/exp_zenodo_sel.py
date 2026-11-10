# =============================================================
# Nombre del archivo: exp_zenodo_sel.py
#
# Descripción:
#   Experimento: robot + solo las 4 vacas de Zenodo con BCS 2.75 / 3.75
#   que quedan en Dataset/Dataset_Zenodo_BCS (z5148_2.75, z711111_2.75,
#   z4891_3.75, z7415_3.75). Zenodo solo se suma al train de cada fold;
#   la evaluación es siempre sobre las vacas del robot (igual que la base).
#
#   Los descriptores salen de <work>/zenodo/features.npz (las 2,088 fotos)
#   filtrados por nombre de carpeta.
#
#   Variantes:  base | +4 zen (todas, 77 fotos) | +4 zen k=5 (5 por vaca)
#
# Uso: docker compose run --rm --entrypoint python bcs exp_zenodo_sel.py
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

SEL = ["z5148_2.75", "z711111_2.75", "z4891_3.75", "z7415_3.75"]

def per_cow(p, y, cows, uc): return np.array([np.abs(p - y)[cows == u].mean() for u in uc])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); work = Path(a.work_dir)
    Xr, yr, cr, nr = br.load_features(work)
    Xz, yz, cz, nz = br.load_features(work / "zenodo")
    folder = np.array([n.split("/")[0] for n in nz])
    iall = np.where(np.isin(folder, SEL))[0]
    ik5 = np.concatenate([g[np.unique(np.linspace(0, len(g) - 1, 5).round().astype(int))]
                          for g in (np.sort(iall[folder[iall] == f]) for f in SEL)])
    print(f"[datos] robot {len(yr)} fotos / {len(np.unique(cr))} vacas | Zenodo sel: "
          + ", ".join(f"{f} ({np.sum(folder == f)})" for f in SEL))

    ucr = np.unique(cr); strata = np.round(yr * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(ucr)).split(yr, yr, cr))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yr, strata, cr)) for s in range(a.seeds)]}
    variants = {"base": None, "+4 zen (77 fotos)": iall, "+4 zen k=5 (20 fotos)": ik5}
    rows, loco = [], {}
    for name, idx in variants.items():
        res = {}
        for sch, reps in schemes.items():
            P = []
            for splits in reps:
                p = np.zeros(len(yr))
                for tr, te in splits:
                    Xtr, ytr, gtr = Xr[tr], yr[tr], cr[tr]
                    if idx is not None:
                        Xtr = np.vstack([Xtr, Xz[idx]]); ytr = np.concatenate([ytr, yz[idx]]); gtr = np.concatenate([gtr, cz[idx]])
                    p[te], _ = br.fit_predict(Xtr, ytr, gtr, Xr[te])
                P.append(p)
            res[sch] = np.mean(P, 0)
        loco[name] = res["loco"]
        for sch, p in res.items():
            m = br.metrics(p, yr, cr); e = p - yr
            rows.append({"variante": name, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_2-3": round(float(e[yr < 3].mean()), 3), "sesgo_3-3.75": round(float(e[(yr >= 3) & (yr < 3.75)].mean()), 3),
                         "sesgo_3.75-4.5": round(float(e[(yr >= 3.75) & (yr < 4.5)].mean()), 3), "sesgo_4.5-5": round(float(e[yr >= 4.5].mean()), 3),
                         "mae_2.75": round(float(np.abs(e[yr == 2.75]).mean()), 3), "mae_3.75": round(float(np.abs(e[yr == 3.75]).mean()), 3)})
            r = rows[-1]
            print(f"{name:22s} {sch:7s} MAE={r['mae']:.3f} R2={r['r2_fotos']:.2f} ±0.5={r['acc_05']:.0%} MAE vaca={r['mae_vaca']:.3f} | "
                  f"MAE 2.75={r['mae_2.75']:.3f} 3.75={r['mae_3.75']:.3f} | sesgo 2-3={r['sesgo_2-3']:+.2f} 3-3.75={r['sesgo_3-3.75']:+.2f} "
                  f"3.75-4.5={r['sesgo_3.75-4.5']:+.2f} 4.5-5={r['sesgo_4.5-5']:+.2f}", flush=True)

    rng = np.random.RandomState(0); ref = per_cow(loco["base"], yr, cr, ucr)
    print("\nDiferencia de MAE por vaca del robot vs base (LOCO, IC95 bootstrap por vaca):")
    for n, p in loco.items():
        if n == "base": continue
        d = per_cow(p, yr, cr, ucr) - ref; bs = [d[rng.randint(0, len(ucr), len(ucr))].mean() for _ in range(5000)]
        print(f"  {n:22s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  mejoran {np.sum(d < 0)}/{len(ucr)}")
        for u in ucr[np.argsort(d)[:3]].tolist() + ucr[np.argsort(d)[-3:]].tolist():
            k = list(ucr).index(u); print(f"      vaca {u} (BCS {yr[cr == u][0]}): {ref[k]:.3f} -> {ref[k] + d[k]:.3f}")

    out = work / "exp_zenodo_sel"; out.mkdir(exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco))
        for i in range(len(yr)): w.writerow([nr[i], cr[i], yr[i]] + [round(float(loco[k][i]), 4) for k in loco])
    print(f"-> {out}")

if __name__ == "__main__":
    main()

# =============================================================
# Nombre del archivo: exp_unict.py
#
# Descripción:
#   Experimento [10]: regresor de BCS con el segmentador nuevo (robot+UNICT,
#   work/seg/final_all40_unict40.pt) y fotos UNICT en el entrenamiento.
#   UNICT solo se suma al train de cada fold; la evaluación es SIEMPRE sobre
#   las vacas del robot (igual que la base y que [7]/[8]).
#
#   Entradas:
#     <work>/features.npz             robot, segmentador actual (base)
#     <work>/robot_seg2/features.npz  robot, segmentador nuevo
#     <work>/unict/features.npz       UNICT (Dataset_UNICT_BCS), segmentador nuevo
#   Variantes:
#     base                 robot, segmentador actual (modelo final de referencia)
#     seg2                 robot, segmentador nuevo
#     seg2 +UNICT          + todas las fotos UNICT (207, 29 vacas)
#     seg2 +UNICT k=3      + 3 fotos por vaca UNICT
#   Referencias: robot(seg2) -> UNICT, y UNICT -> UNICT (LOCO por vaca).
#
# Uso: python exp_unict.py --work_dir /work
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

def per_cow(p, y, cows, uc): return np.array([np.abs(p - y)[cows == u].mean() for u in uc])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args(); work = Path(a.work_dir)
    X0, y0, c0, n0 = br.load_features(work)
    Xr, yr, cr, nr = br.load_features(work / "robot_seg2")
    Xu, yu, cu, nu = br.load_features(work / "unict")
    assert list(n0) == list(nr), "robot: las fotos de base y seg2 no coinciden"
    k3 = np.concatenate([g[np.unique(np.linspace(0, len(g) - 1, 3).round().astype(int))]
                         for g in (np.where(cu == u)[0] for u in np.unique(cu))])
    print(f"[datos] robot {len(yr)} fotos / {len(np.unique(cr))} vacas | UNICT {len(yu)} fotos / {len(np.unique(cu))} vacas "
          f"(BCS {yu.min():g}-{yu.max():g}, sd por vaca {np.std([yu[cu == u][0] for u in np.unique(cu)]):.2f})")

    ucr = np.unique(cr); strata = np.round(yr * 2).astype(int)
    schemes = {"loco": [list(GroupKFold(len(ucr)).split(yr, yr, cr))],
               "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(yr, strata, cr)) for s in range(a.seeds)]}
    variants = {"base": (X0, None), "seg2": (Xr, None), "seg2 +UNICT": (Xr, np.arange(len(yu))), "seg2 +UNICT k=3": (Xr, k3)}
    rows, loco = [], {}
    for name, (XR, idx) in variants.items():
        res = {}
        for sch, reps in schemes.items():
            P = []
            for splits in reps:
                p = np.zeros(len(yr))
                for tr, te in splits:
                    Xtr, ytr, gtr = XR[tr], yr[tr], cr[tr]
                    if idx is not None:
                        Xtr = np.vstack([Xtr, Xu[idx]]); ytr = np.concatenate([ytr, yu[idx]]); gtr = np.concatenate([gtr, cu[idx]])
                    p[te], _ = br.fit_predict(Xtr, ytr, gtr, XR[te])
                P.append(p)
            res[sch] = np.mean(P, 0)
        loco[name] = res["loco"]
        for sch, p in res.items():
            m = br.metrics(p, yr, cr); e = p - yr
            rows.append({"variante": name, "esquema": sch, **{k: round(float(v), 4) for k, v in m.items()},
                         "sesgo_2-3": round(float(e[yr < 3].mean()), 3), "sesgo_3-3.75": round(float(e[(yr >= 3) & (yr < 3.75)].mean()), 3),
                         "sesgo_3.75-4.5": round(float(e[(yr >= 3.75) & (yr < 4.5)].mean()), 3), "sesgo_4.5-5": round(float(e[yr >= 4.5].mean()), 3)})
            r = rows[-1]
            print(f"{name:17s} {sch:7s} MAE={r['mae']:.3f} R2={r['r2_fotos']:.2f} ±0.5={r['acc_05']:.0%} MAE vaca={r['mae_vaca']:.3f} | "
                  f"sesgo 2-3={r['sesgo_2-3']:+.2f} 3-3.75={r['sesgo_3-3.75']:+.2f} 3.75-4.5={r['sesgo_3.75-4.5']:+.2f} 4.5-5={r['sesgo_4.5-5']:+.2f}", flush=True)

    rng = np.random.RandomState(0)
    for refname in ["base", "seg2"]:
        ref = per_cow(loco[refname], yr, cr, ucr)
        print(f"\nDiferencia de MAE por vaca del robot vs {refname} (LOCO, IC95 bootstrap por vaca):")
        for n, p in loco.items():
            if n == refname or (refname == "seg2" and n == "base"): continue
            d = per_cow(p, yr, cr, ucr) - ref; bs = [d[rng.randint(0, len(ucr), len(ucr))].mean() for _ in range(5000)]
            print(f"  {n:17s} {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  mejoran {np.sum(d < 0)}/{len(ucr)}")

    # referencias en UNICT
    pu, _ = br.fit_predict(Xr, yr, cr, Xu)
    print(f"\nrobot(seg2) -> UNICT: MAE={np.abs(pu - yu).mean():.3f} sesgo={np.mean(pu - yu):+.2f}")
    ucu = np.unique(cu); pl = np.zeros(len(yu))
    for tr, te in GroupKFold(len(ucu)).split(yu, yu, cu): pl[te], _ = br.fit_predict(Xu[tr], yu[tr], cu[tr], Xu[te])
    print(f"UNICT -> UNICT (LOCO): MAE={np.abs(pl - yu).mean():.3f} R2={br.r2(pl, yu):.2f} | predecir la media: MAE={np.abs(yu - yu.mean()).mean():.3f}")

    out = work / "exp_unict"; out.mkdir(exist_ok=True)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out / "preds_loco.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real"] + list(loco))
        for i in range(len(yr)): w.writerow([nr[i], cr[i], yr[i]] + [round(float(loco[k][i]), 4) for k in loco])
    print(f"-> {out}")

if __name__ == "__main__":
    main()

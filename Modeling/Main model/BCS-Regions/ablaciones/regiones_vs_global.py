# =============================================================
# Nombre del archivo: ablaciones/regiones_vs_global.py
#
# Descripción:
#   ¿Aportan las regiones anatómicas frente a describir la foto completa
#   o un solo recorte? Mismo backbone (DINOv3-L/16 congelado, 384 px, bloque br.LAYER),
#   mismo ridge del modelo (comun.fit_apply = bcs_regions.fit_predict) y misma
#   validación por vaca (LOCO y 5 folds x semillas). Variantes:
#     regiones          el modelo base: promedio por cada una de las 6 regiones + ausencias (features.npz)
#     foto_completa     token CLS + promedio de todos los parches de la foto
#     vaca              promedio de los parches dentro de la silueta (unión de las 6 máscaras)
#     recorte_trasero   la foto recortada al rectángulo de cadera+grupa+cola+muslos, CLS + promedio
#                       (lo más parecido a los métodos de un solo recorte, p. ej. Nagy et al. 2023)
#     regiones+completa concatenación de regiones y foto_completa
#   Comparación pareada contra 'regiones': diferencia de MAE por vaca×BCS
#   (LOCO) con IC 95% por bootstrap de vacas.
#   Salidas: <work>/ablaciones/{feat_<variante>.npz, resumen.csv, preds_<variante>_<esquema>.csv}
#
# Uso (GPU solo la primera vez, para los descriptores):
#   docker compose run --rm --entrypoint python bcs ablaciones/regiones_vs_global.py
# =============================================================
import argparse, csv, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from comun import splits_for, fit_apply, tokens, grid
from data import load_records
from segmenter import REGIONS

TRASERO = [REGIONS.index(r) for r in ("hips", "rump", "tailhead", "thigh_L", "thigh_R")]

def compute(work, recs, out):
    dev = br.device(); bb = br.load_backbone(dev); F = {"foto_completa": [], "vaca": [], "recorte_trasero": []}
    g = grid(bb)
    for i, r in enumerate(recs):
        img = br.read_rgb(r["path"]); m = np.load(br.mask_file(work, r))["masks"].astype(np.float32) / 255.0
        cls, pt = tokens(bb, img, dev)
        F["foto_completa"].append(torch.cat([cls, pt.mean(0)]).cpu().numpy())
        u = torch.from_numpy((m.max(0) > 0.5).astype(np.float32))[None, None]
        w = torch.nn.functional.adaptive_avg_pool2d(torch.nn.functional.interpolate(u, size=(br.SIZE, br.SIZE), mode="nearest"), (g, g)).reshape(-1).to(dev)
        F["vaca"].append(((w @ pt) / w.sum().clamp_min(1e-6)).cpu().numpy())
        ys, xs = np.where(m[TRASERO].max(0) > 0.5); sy, sx = img.shape[0] / m.shape[1], img.shape[1] / m.shape[2]
        crop = img[int(ys.min() * sy):int((ys.max() + 1) * sy), int(xs.min() * sx):int((xs.max() + 1) * sx)] if len(ys) else img
        c2, p2 = tokens(bb, crop, dev); F["recorte_trasero"].append(torch.cat([c2, p2.mean(0)]).cpu().numpy())
        if (i + 1) % 100 == 0: print(f"  {i + 1}/{len(recs)}", flush=True)
    for k, v in F.items(): np.savez_compressed(out / f"feat_{k}.npz", X=np.stack(v).astype(np.float32), name=np.array([r["name"] for r in recs]))

def cow_err(p, y, cows):
    keys = sorted(set(zip(cows, y)))
    return np.array([abs(p[(cows == c) & (y == b)].mean() - b) for c, b in keys])

def main(a):
    work = Path(a.work_dir); out = work / "ablaciones"; out.mkdir(exist_ok=True)
    X, y, cows, names = br.load_features(work); recs = load_records(a.data_dir)
    assert [r["name"] for r in recs] == list(names), "features.npz no coincide con los datos: corre bcs_regions.py features"
    if not all((out / f"feat_{k}.npz").exists() for k in ("foto_completa", "vaca", "recorte_trasero")): compute(work, recs, out)
    V = {"regiones": X}
    for k in ("foto_completa", "vaca", "recorte_trasero"): V[k] = np.load(out / f"feat_{k}.npz")["X"].astype(np.float64)
    V["regiones+completa"] = np.hstack([X, V["foto_completa"]])
    schemes = splits_for(y, cows, a.seeds); rows, ce = [], {}
    for v, Xv in V.items():
        t0 = time.time()
        for sch, reps in schemes.items():
            P = []
            for splits in reps:
                p = np.zeros(len(y))
                for tr, te in splits: p[te] = fit_apply(Xv[tr], y[tr], cows[tr], Xv[te])
                P.append(p)
            P = np.mean(P, 0); m = br.metrics(P, y, cows)
            if sch == "loco": ce[v] = cow_err(P, y, cows)
            rows.append({"variante": v, "esquema": sch, "dim": Xv.shape[1], **{k: round(float(x), 4) for k, x in m.items()}})
            with open(out / f"preds_{v}_{sch}.csv", "w", newline="") as fh:
                w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
                w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, P)])
        print(f"{v}: {time.time() - t0:.0f}s", flush=True)
    rng = np.random.RandomState(0); n = len(ce["regiones"])
    for r in rows:
        d = ce[r["variante"]] - ce["regiones"]; bs = [d[rng.randint(0, n, n)].mean() for _ in range(5000)]
        r["dif_mae_vs_regiones_loco"] = round(float(d.mean()), 4); r["dif_ic95_lo"], r["dif_ic95_hi"] = [round(float(q), 4) for q in np.percentile(bs, [2.5, 97.5])]
    with open(out / "resumen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n{'variante':18s} {'esquema':8s} {'dim':>5s} {'MAE':>6s} {'R2fot':>6s} {'R2vac':>6s} {'MAEvaca':>7s}  dif. vs regiones (LOCO, por vaca×BCS) [IC95]")
    for r in rows:
        print(f"{r['variante']:18s} {r['esquema']:8s} {r['dim']:5d} {r['mae']:6.3f} {r['r2_fotos']:6.3f} {r['r2_vacas']:6.3f} {r['mae_vaca']:7.3f}  "
              f"{r['dif_mae_vs_regiones_loco']:+.3f} [{r['dif_ic95_lo']:+.3f}, {r['dif_ic95_hi']:+.3f}]")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

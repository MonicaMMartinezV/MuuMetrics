# =============================================================
# Nombre del archivo: baselines/comparar.py
#
# Descripción:
#   Junta las predicciones fuera de fold de todos los baselines
#   (<work>/baselines/<variante>_<division>/preds_seed<s>.csv, promedio
#   sobre semillas) y las compara contra el modelo (bcs_regions_<division>)
#   en la misma división:
#     MAE, RMSE, R² por foto y por vaca, exactitud a ±0.25 y ±0.5 (BCS
#     redondeado a 0.25, como reporta la literatura), MAE por vaca con IC95,
#     sesgo en flacas (< 3) y gordas (>= 4.5), y diferencia pareada de MAE
#     por vaca contra el modelo con IC95 (bootstrap de vacas).
#   La tabla "foto" muestra cuánto se infla cada método con fuga de
#   identidad (la misma vaca en train y test).
#   Salida: <work>/baselines/comparacion.csv
#
# Uso: docker compose -f baselines/docker-compose.yml run --rm baselines baselines/comparar.py
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np

from comun_b import br, cargar, leer_preds, DIVISIONES

def main(a):
    base = Path(a.work_dir) / "baselines"; _, y, cows, names = cargar(a.data_dir); uc = np.unique(cows)
    rng = np.random.RandomState(0); idx_b = [rng.randint(0, len(uc), len(uc)) for _ in range(5000)]; rows = []
    for div in DIVISIONES:
        P = {}
        for d in sorted(base.glob(f"*_{div}")):
            fs = sorted(d.glob("preds_seed*.csv"))
            if not fs: continue
            preds = []
            for f in fs:
                n, p = leer_preds(f); assert list(n) == list(names), f; preds.append(p)
            P[d.name[: -len(div) - 1]] = (np.mean(preds, 0), len(fs))
        if "bcs_regions" not in P: print(f"[{div}] falta bcs_regions (corre congelados.py)"); continue
        ref = np.array([np.abs(P["bcs_regions"][0] - y)[cows == c].mean() for c in uc])
        print(f"\n=== división por {div} ===")
        print(f"{'variante':24s} {'sem':>3s} {'MAE':>6s} {'RMSE':>6s} {'R2fot':>6s} {'R2vac':>6s} {'±.25':>5s} {'±.5':>5s} {'MAEvaca':>7s} "
              f"{'flacas':>7s} {'gordas':>7s}  dif. MAE vaca vs modelo [IC95]")
        for v, (p, ns) in sorted(P.items(), key=lambda kv: (kv[0] != "bcs_regions", np.abs(kv[1][0] - y).mean())):
            m = br.metrics(p, y, cows); rd = np.clip(np.round(p / 0.25) * 0.25, 2.0, 5.0); e = p - y
            dv = np.array([np.abs(p - y)[cows == c].mean() for c in uc]) - ref
            lo, hi = np.percentile([dv[i].mean() for i in idx_b], [2.5, 97.5])
            r = {"division": div, "variante": v, "semillas": ns, **{k: round(float(x), 4) for k, x in m.items()},
                 "acc_025": round(float(np.mean(np.abs(rd - y) <= 0.25 + 1e-9)), 4),
                 "sesgo_flacas": round(float(e[y < 3].mean()), 3), "sesgo_gordas": round(float(e[y >= 4.5].mean()), 3),
                 "dif_mae_vaca": round(float(dv.mean()), 4), "dif_ic95_lo": round(float(lo), 4), "dif_ic95_hi": round(float(hi), 4)}
            rows.append(r)
            print(f"{v:24s} {ns:3d} {r['mae']:6.3f} {r['rmse']:6.3f} {r['r2_fotos']:6.2f} {r['r2_vacas']:6.2f} {r['acc_025']:5.0%} {r['acc_05']:5.0%} "
                  f"{r['mae_vaca']:7.3f} {r['sesgo_flacas']:+7.2f} {r['sesgo_gordas']:+7.2f}  "
                  + ("—" if v == "bcs_regions" else f"{dv.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"))
    if rows:
        with open(base / "comparacion.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\n-> {base / 'comparacion.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())

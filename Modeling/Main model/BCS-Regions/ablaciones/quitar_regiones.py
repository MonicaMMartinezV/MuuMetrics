# =============================================================
# Nombre del archivo: ablaciones/quitar_regiones.py
#
# Descripción:
#   ¿Mejora el modelo base si se quitan las regiones que le restan?
#   Usa <work>/features.npz (regiones y dimensión según su firma; hoy 6 x 1024 + 6); solo CPU.
#   Mismo ridge del modelo (comun.fit_apply = bcs_regions.fit_predict) y validación por vaca.
#     1. Quitar una región a la vez (LOCO y 5x10), con la diferencia de MAE por
#        vaca×BCS contra el modelo completo e IC 95% (bootstrap de vacas).
#     2. 'sin_las_que_restan': quitar todas las regiones cuya ausencia bajó el MAE
#        en el paso 1. OJO: se eligen con la misma validación -> estimación optimista.
#     3. 'seleccion_anidada': eliminación hacia atrás (greedy) DENTRO del train de
#        cada fold LOCO, con GroupKFold(5) por vaca; es la estimación honesta de
#        "quitar lo que resta". Reporta qué regiones se quitaron en cuántos folds.
#   Salidas: <work>/ablaciones/quitar_regiones.csv y seleccion_anidada.csv
#
# Uso: docker compose run --rm --entrypoint python bcs ablaciones/quitar_regiones.py
# =============================================================
import argparse, csv, sys, time
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from comun import splits_for, fit_apply, modelo

ES = {"loin": "lomo", "hips": "cadera", "rump": "grupa", "tailhead": "base_cola", "thigh_L": "muslo_izq", "thigh_R": "muslo_der"}
REG, D = None, None               # regiones del descriptor y dimensión: salen de la firma de features.npz (main)

def cols(keep):
    idx = [REG.index(r) for r in keep]
    return np.concatenate([np.arange(i * D, (i + 1) * D) for i in idx] + [len(REG) * D + np.array(idx)])

def run(X, y, cows, splits):
    p = np.zeros(len(y))
    for tr, te in splits: p[te] = fit_apply(X[tr], y[tr], cows[tr], X[te])
    return p

def cow_err(p, y, cows):
    return np.array([abs(p[(cows == c) & (y == b)].mean() - b) for c, b in sorted(set(zip(cows, y)))])

def inner_mae(X, y, cows, keep):
    Xs = X[:, cols(keep)]; e = []
    for tr, te in GroupKFold(5).split(Xs, y, cows):
        e.append(np.abs(fit_apply(Xs[tr], y[tr], cows[tr], Xs[te]) - y[te]))
    return np.concatenate(e).mean()

def backward(X, y, cows):
    keep = list(REG); best = inner_mae(X, y, cows, keep)
    while len(keep) > 1:
        cand = {r: inner_mae(X, y, cows, [k for k in keep if k != r]) for r in keep}
        r, m = min(cand.items(), key=lambda kv: kv[1])
        if m >= best: break
        keep.remove(r); best = m
    return keep

def main(a):
    work = Path(a.work_dir); out = work / "ablaciones"; out.mkdir(exist_ok=True)
    global REG, D
    X, y, cows, _, _, REG, D = modelo(work); schemes = splits_for(y, cows, a.seeds)
    rng = np.random.RandomState(0); rows = []; base_ce = None

    def evaluate(nombre, keep, nota=""):
        nonlocal base_ce
        Xs = X[:, cols(keep)]; r = {"variante": nombre, "regiones": "+".join(ES[k] for k in keep), "nota": nota}
        for sch, reps in schemes.items():
            P = np.mean([run(Xs, y, cows, s) for s in reps], 0); m = br.metrics(P, y, cows)
            r.update({f"mae_{sch}": round(float(m["mae"]), 4), f"r2_fotos_{sch}": round(float(m["r2_fotos"]), 4),
                      f"r2_vacas_{sch}": round(float(m["r2_vacas"]), 4)})
            if sch == "loco":
                ce = cow_err(P, y, cows)
                if base_ce is None: base_ce = ce
                d = ce - base_ce; bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
                r.update({"dif_vs_completo": round(float(d.mean()), 4),
                          "ic95_lo": round(float(np.percentile(bs, 2.5)), 4), "ic95_hi": round(float(np.percentile(bs, 97.5)), 4)})
        rows.append(r); print(f"  {nombre:22s} MAE loco {r['mae_loco']:.3f} 5x10 {r['mae_rep5x10']:.3f}  dif {r['dif_vs_completo']:+.3f} "
                              f"[{r['ic95_lo']:+.3f}, {r['ic95_hi']:+.3f}]", flush=True)
        return r

    t0 = time.time(); print("[1] quitar una región")
    full = evaluate("completo", REG)
    restan = []
    for reg in REG:
        r = evaluate(f"sin_{ES[reg]}", [k for k in REG if k != reg])
        if r["mae_rep5x10"] < full["mae_rep5x10"] and r["mae_loco"] < full["mae_loco"]: restan.append(reg)
    print(f"[2] regiones que restan (quitarlas baja el MAE en LOCO y en 5x10): {[ES[r] for r in restan] or 'ninguna'}")
    if restan:
        evaluate("sin_las_que_restan", [k for k in REG if k not in restan], "optimista: elegidas con la misma validación")

    print("[3] selección anidada (eliminación hacia atrás dentro de cada fold LOCO)")
    p = np.zeros(len(y)); quitadas = Counter(); sel = []
    for i, (tr, te) in enumerate(schemes["loco"][0]):
        keep = backward(X[tr], y[tr], cows[tr]); Xs = X[:, cols(keep)]
        p[te] = fit_apply(Xs[tr], y[tr], cows[tr], Xs[te])
        quitadas.update(set(REG) - set(keep)); sel.append({"vaca_test": cows[te][0], "regiones": "+".join(ES[k] for k in keep)})
        if (i + 1) % 8 == 0: print(f"  fold {i + 1}/{len(schemes['loco'][0])} ({time.time() - t0:.0f}s)", flush=True)
    m = br.metrics(p, y, cows); ce = cow_err(p, y, cows); d = ce - base_ce
    bs = [d[rng.randint(0, len(d), len(d))].mean() for _ in range(5000)]
    rows.append({"variante": "seleccion_anidada", "regiones": "varía por fold", "nota": "honesta (solo LOCO)",
                 "mae_loco": round(float(m["mae"]), 4), "r2_fotos_loco": round(float(m["r2_fotos"]), 4), "r2_vacas_loco": round(float(m["r2_vacas"]), 4),
                 "dif_vs_completo": round(float(d.mean()), 4), "ic95_lo": round(float(np.percentile(bs, 2.5)), 4), "ic95_hi": round(float(np.percentile(bs, 97.5)), 4)})
    n = len(schemes["loco"][0])
    print(f"  seleccion_anidada      MAE loco {m['mae']:.3f}  dif {d.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print("  veces que se quitó cada región: " + ", ".join(f"{ES[r]} {quitadas[r]}/{n}" for r in REG))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "quitar_regiones.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(out / "seleccion_anidada.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["vaca_test", "regiones"]); w.writeheader(); w.writerows(sel)
    print(f"-> {out / 'quitar_regiones.csv'} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

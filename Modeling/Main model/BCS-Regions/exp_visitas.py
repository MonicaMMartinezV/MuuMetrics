# =============================================================
# Experimento [23]: ¿mejora promediar varias fotos? Usa las predicciones
# LOCO del modelo actual (work/preds_loco.csv), sin reentrenar.
#   visita = fotos de la misma vaca separadas < GAP min (misma pasada)
#   día    = fotos de la misma vaca el mismo día
#   k fotos al azar de la vaca (cualquier fecha), k = 1, 2, 3, 5, 10
# Uso: python3 exp_visitas.py
# =============================================================
import csv, re
from datetime import datetime
from pathlib import Path
import numpy as np

GAP = 10
R = list(csv.DictReader(open("work/preds_loco.csv"))); rng = np.random.default_rng(0)
for r in R:
    r["t"] = datetime.strptime(re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", r["foto"]).group(), "%Y-%m-%d-%H-%M-%S")
    r["p"], r["y"] = float(r["bcs_pred"]), float(r["bcs_real"])
    r["vaca"] = f'{r["vaca"]}_{r["bcs_real"]}'   # 8758 y 9008 tienen fotos con distinta calificación: unidad = vaca + BCS
cows = sorted({r["vaca"] for r in R}); Y = {c: next(r["y"] for r in R if r["vaca"] == c) for c in cows}

def groups(key):
    G = {}
    for c in cows:
        rs = sorted([r for r in R if r["vaca"] == c], key=lambda r: r["t"]); gi = 0
        for i, r in enumerate(rs):
            if i and key(rs[i - 1], r): gi += 1
            G.setdefault((c, gi), []).append(r["p"])
    return G

def report(name, G):
    e = np.array([np.mean(v) - Y[c] for (c, _), v in G.items()]); n = np.array([len(v) for v in G.values()])
    cw = np.array([np.mean([abs(np.mean(v) - Y[c]) for (cc, _), v in G.items() if cc == c]) for c in cows])
    print(f"{name:26s} grupos={len(G):4d}  fotos/grupo={n.mean():4.1f} (máx {n.max():2d})  MAE={np.abs(e).mean():.3f}  MAE por vaca={cw.mean():.3f}  ±0.5={np.mean(np.abs(np.round(np.array([np.mean(v) for v in G.values()]) * 4) / 4 - np.array([Y[c] for c, _ in G])) <= 0.5):.0%}")

report("1 foto (actual)", {(r["vaca"], i): [r["p"]] for i, r in enumerate(R)})
report(f"visita (< {GAP} min)", groups(lambda a, b: (b["t"] - a["t"]).total_seconds() > GAP * 60))
report("mismo día", groups(lambda a, b: a["t"].date() != b["t"].date()))
print("\nk fotos al azar de la misma vaca (vacas con >= k fotos, 200 sorteos):")
for k in [1, 2, 3, 5, 10]:
    cs = [c for c in cows if sum(r["vaca"] == c for r in R) >= k]; errs = []
    for c in cs:
        ps = np.array([r["p"] for r in R if r["vaca"] == c])
        errs.append(np.mean([abs(ps[rng.choice(len(ps), k, replace=False)].mean() - Y[c]) for _ in range(200)]))
    print(f"  k={k:2d}  vacas={len(cs):2d}  MAE por vaca={np.mean(errs):.3f}")
print("\nreferencia: MAE por vaca usando TODAS sus fotos =",
      f"{np.mean([abs(np.mean([r['p'] for r in R if r['vaca'] == c]) - Y[c]) for c in cows]):.3f}")

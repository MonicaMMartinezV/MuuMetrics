# =============================================================
# Experimento [21]: ¿el modelo actual (224 px / bloque 17) tiene sesgo por
# color de pelaje? No hay etiqueta de pelaje en el dataset; se etiquetó a
# mano viendo 3 fotos por vaca (zona dorsal: lomo, cadera, grupa):
#   blanca = casi todo blanco, manchas pequeñas
#   negra  = casi todo negro, sólo una mancha blanca pequeña
#   pinta  = mezcla
# -> work/exp_pelaje/pelaje_vacas.csv (corregible a mano).
# (Medir el negro por brillo de píxel no sirvió: la iluminación del robot
#  cambia mucho y la sombra de la grupa se confunde con pelo negro.)
#
# Se cruza con work/preds_loco.csv: error (pred - real) por vaca, por grupo,
# IC95 por bootstrap de vacas, y una regresión por vaca
#   error ~ grupo + BCS real
# porque el ridge sobreestima a las flacas y subestima a las gordas: si un
# grupo tiene más flacas parecería sesgado sin serlo.
# Uso: python3 exp_pelaje.py
# =============================================================
import csv
from pathlib import Path
import numpy as np

W = Path("work"); rng = np.random.default_rng(0)
pel = {r["vaca"]: r["pelaje"] for r in csv.DictReader(open(W / "exp_pelaje/pelaje_vacas.csv"))}
rows = list(csv.DictReader(open(W / "preds_loco.csv")))
cows = sorted({r["vaca"] for r in rows})
E = {c: np.array([float(r["bcs_pred"]) - float(r["bcs_real"]) for r in rows if r["vaca"] == c]) for c in cows}
Y = {c: np.mean([float(r["bcs_real"]) for r in rows if r["vaca"] == c]) for c in cows}   # 8758 y 9008 tienen 2-3 calificaciones
G = ["blanca", "pinta", "negra"]

def boot(v, f=np.mean, n=5000):
    v = np.asarray(v); b = [f(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])

print(f"{'grupo':7s} {'vacas':>5s} {'fotos':>5s} {'BCS medio':>9s} {'flacas<3':>8s} {'sesgo (pred-real)':>26s} {'MAE':>22s}")
out = []
for g in G:
    cs = [c for c in cows if pel[c] == g]
    b = [E[c].mean() for c in cs]; m = [np.abs(E[c]).mean() for c in cs]
    lb, ub = boot(b); lm, um = boot(m)
    print(f"{g:7s} {len(cs):5d} {sum(len(E[c]) for c in cs):5d} {np.mean([Y[c] for c in cs]):9.2f} {sum(Y[c] < 3 for c in cs):8d}"
          f"   {np.mean(b):+.3f} [{lb:+.3f}, {ub:+.3f}]   {np.mean(m):.3f} [{lm:.3f}, {um:.3f}]")
    out.append({"grupo": g, "vacas": len(cs), "bcs_medio": round(np.mean([Y[c] for c in cs]), 3),
                "sesgo": round(np.mean(b), 4), "sesgo_ic": f"[{lb:+.3f}, {ub:+.3f}]", "mae": round(np.mean(m), 4), "mae_ic": f"[{lm:.3f}, {um:.3f}]"})

# regresión por vaca: sesgo ~ BCS real + blanca + negra (pinta = referencia); IC por bootstrap de vacas
def fit(cs, target):
    X = np.array([[1, Y[c] - 3.25, pel[c] == "blanca", pel[c] == "negra"] for c in cs], float)
    t = np.array([target(c) for c in cs]); return np.linalg.lstsq(X, t, rcond=None)[0]
for name, tgt in [("sesgo", lambda c: E[c].mean()), ("MAE", lambda c: np.abs(E[c]).mean())]:
    beta = fit(cows, tgt); B = np.array([fit(list(np.array(cows)[rng.integers(0, len(cows), len(cows))]), tgt) for _ in range(5000)])
    lo, hi = np.percentile(B, [2.5, 97.5], 0)
    print(f"\n{name} ajustado por BCS real (contra pintas):")
    for k, n in [(1, "por punto de BCS"), (2, "blanca - pinta"), (3, "negra - pinta")]:
        print(f"  {n:17s} {beta[k]:+.3f} [{lo[k]:+.3f}, {hi[k]:+.3f}]")
    out.append({"grupo": f"ajustado_{name}", "sesgo": f"blanca {beta[2]:+.3f} [{lo[2]:+.3f},{hi[2]:+.3f}] | negra {beta[3]:+.3f} [{lo[3]:+.3f},{hi[3]:+.3f}]"})

# por rango de BCS dentro de cada grupo
print("\nsesgo por rango de BCS (vacas):")
for lo_, hi_, n in [(0, 3, "flacas <3"), (3, 3.75, "3-3.5"), (3.75, 9, "gordas >=3.75")]:
    print(f"  {n:14s} " + "  ".join(f"{g}: {np.mean([E[c].mean() for c in cows if pel[c]==g and lo_<=Y[c]<hi_]) if any(pel[c]==g and lo_<=Y[c]<hi_ for c in cows) else float('nan'):+.2f} (n={sum(pel[c]==g and lo_<=Y[c]<hi_ for c in cows)})" for g in G))
with open(W / "exp_pelaje/resumen.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["grupo", "vacas", "bcs_medio", "sesgo", "sesgo_ic", "mae", "mae_ic"]); w.writeheader(); w.writerows(out)

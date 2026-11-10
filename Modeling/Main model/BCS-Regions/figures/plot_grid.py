# Figuras del grid search (EXPERIMENTOS [19]) a partir de work/exp_grid/grid.csv.
# Uso: python3 figures/plot_grid.py   -> figures/grid_bloque_resolucion.png, figures/grid_alpha.png
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
rows = list(csv.DictReader(open(ROOT / "work/exp_grid/grid.csv")))
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]   # secuencial, claro -> oscuro
S1, S2 = "#2a78d6", "#eb6834"                                                        # categórico slots 1 y 2
CUR_RES, CUR_BLK = 378, 18
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})

def get(res, blk, alpha, sch, key):
    for r in rows:
        if int(r["res"]) == res and int(r["bloque"]) == blk and r["alpha"] == alpha and r["esquema"] == sch:
            return float(r[key])

RES = sorted({int(r["res"]) for r in rows}); BLK = sorted({int(r["bloque"]) for r in rows})

# ---- 1. Heatmap R2 (LOCO, alpha interno) por resolución x bloque ----
fig, axes = plt.subplots(2, 1, figsize=(11, 5.4), constrained_layout=True)
for ax, (key, lab, fmt) in zip(axes, [("r2_fotos", "R² por foto", "{:.2f}"), ("mae", "MAE", "{:.3f}")]):
    M = np.array([[get(r, b, "interno", "loco", key) for b in BLK] for r in RES])
    better = M if key == "r2_fotos" else -M                     # más oscuro = mejor en ambos paneles
    cmap = LinearSegmentedColormap.from_list("azul", BLUE)
    ax.imshow(better, cmap=cmap, aspect="auto")
    lo, hi = better.min(), better.max()
    for i in range(len(RES)):
        for j in range(len(BLK)):
            t = (better[i, j] - lo) / max(hi - lo, 1e-9)
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5, color="white" if t > 0.55 else INK)
    ci, cj = RES.index(CUR_RES), BLK.index(CUR_BLK)
    ax.add_patch(Rectangle((cj - .5, ci - .5), 1, 1, fill=False, ec=S2, lw=2.2))
    ax.set_xticks(range(len(BLK)), [str(b) for b in BLK]); ax.set_yticks(range(len(RES)), [f"{r} px" for r in RES])
    ax.set_title(f"{lab} (LOCO, 64 vacas) · más oscuro = mejor · recuadro = modelo actual", loc="left", color=INK, fontsize=10.5)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[-1].set_xlabel("bloque de DINOv2-L (de 24)")
fig.suptitle("Grid search BCS-Regions: resolución de entrada × bloque (alpha elegido por vaca dentro de cada fold)",
             x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
fig.savefig(ROOT / "figures/grid_bloque_resolucion.png", dpi=160)

# ---- 2. MAE vs alpha en la configuración actual ----
ALPHAS = [a for a in dict.fromkeys(r["alpha"] for r in rows) if a != "interno"]
x = [float(a) for a in ALPHAS]
fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
for sch, col, name, dy in [("loco", S1, "LOCO", -9), ("rep5x10", S2, "5 folds × 10", 9)]:
    y = [get(CUR_RES, CUR_BLK, a, sch, "mae") for a in ALPHAS]
    ax.plot(x, y, color=col, lw=2, marker="o", ms=6, mec=SURF, mew=2, label=name)
    yi = get(CUR_RES, CUR_BLK, "interno", sch, "mae")
    ax.axhline(yi, color=col, lw=1, ls=(0, (3, 3)))
    ax.annotate(f"alpha elegido por vaca: {yi:.3f}", (x[-1] * 1.6, yi), textcoords="offset points", xytext=(0, dy),
                va="center", ha="left", color=INK2, fontsize=8.5)
ax.set_xscale("log"); ax.set_xlabel("alpha del ridge (fijo)"); ax.set_ylabel("MAE")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
ax.set_title(f"MAE contra alpha · {CUR_RES} px, bloque {CUR_BLK} (modelo actual)", loc="left", color=INK, fontsize=11)
ax.set_xlim(x[0] / 2, x[-1] * 60); ax.set_ylim(0.47, None)
fig.savefig(ROOT / "figures/grid_alpha.png", dpi=160)
print("-> figures/grid_bloque_resolucion.png, figures/grid_alpha.png")

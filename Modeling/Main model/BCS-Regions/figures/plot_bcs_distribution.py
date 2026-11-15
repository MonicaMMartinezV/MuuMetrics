"""Distribución de BCS por vaca en el dataset (figura del paper).

(a) Número de vacas por valor de BCS.
(b) Una barra por vaca, ordenadas por BCS: número de imágenes de cada vaca.
Las vacas calificadas más de una vez (8758, 9008) cuentan una vez por cada BCS.

Uso:
  python figures/plot_bcs_distribution.py --data_dir ../../../Dataset/Dataset_Fr_BCS_v2
"""
import argparse, re
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="../../../Dataset/Dataset_Fr_BCS_v2")
ap.add_argument("--out", default=str(Path(__file__).parent / "bcs_distribution"))
a = ap.parse_args()

EXT = {".jpg", ".jpeg", ".png"}
rows = []  # (vaca, bcs, n_imagenes)
for d in sorted(Path(a.data_dir).iterdir()):
    m = re.fullmatch(r"(\d+)_(\d+(?:\.\d+)?)", d.name)
    if not d.is_dir() or not m:
        continue
    n = sum(1 for p in d.iterdir() if p.suffix.lower() in EXT)
    if n:
        rows.append((m.group(1), float(m.group(2)), n))
rows.sort(key=lambda r: (r[1], -r[2], r[0]))
cows = {r[0] for r in rows}
print(f"{len(cows)} vacas, {len(rows)} pares vaca-BCS, {sum(r[2] for r in rows)} imágenes")

INK, INK2, GRID, BAR = "#1f1f1f", "#555555", "#d9d9d9", "#2c6e9e"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

levels = np.arange(2.0, 5.0001, 0.25)
cnt = Counter(r[1] for r in rows)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw={"width_ratios": [1, 1.9], "wspace": 0.22})

# (a) vacas por BCS
y = [cnt.get(l, 0) for l in levels]
ax1.bar(levels, y, width=0.2, color=BAR, edgecolor="white", linewidth=0.5, zorder=2)
for l, v in zip(levels, y):
    if v: ax1.text(l, v + 0.15, str(v), ha="center", va="bottom", fontsize=7, color=INK)
ax1.set_xticks(levels[::2]); ax1.set_xlim(1.8, 5.2)
ax1.set_ylim(0, max(y) * 1.18)
ax1.set_xlabel("BCS"); ax1.set_ylabel("Number of cows")
ax1.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax1.set_title("(a)", loc="left", fontweight="bold", color=INK)

# (b) imágenes por vaca, agrupadas por BCS
x, gap, pos, centers = 0.0, 1.0, [], {}
prev = None
for _, b, _ in rows:
    if prev is not None and b != prev: x += gap
    pos.append(x); centers.setdefault(b, []).append(x); x += 1; prev = b
n_img = [r[2] for r in rows]
ax2.bar(pos, n_img, width=0.8, color=BAR, linewidth=0, zorder=2)
ax2.set_xticks([np.mean(v) for v in centers.values()])
ax2.set_xticklabels([f"{b:.2f}" for b in centers], fontsize=6.5, rotation=90)
ax2.tick_params(axis="x", length=0)
for i, v in enumerate(centers.values()):  # bandas alternas para separar los grupos de BCS
    if i % 2 == 0:
        ax2.axvspan(v[0] - 0.5 - gap / 2, v[-1] + 0.5 + gap / 2, color="#f2f2f2", lw=0, zorder=0)
ax2.set_xlim(-1.5, pos[-1] + 1.5)
ax2.set_xlabel("BCS (one bar per cow)"); ax2.set_ylabel("Images per cow")
ax2.grid(axis="y", color=GRID, lw=0.5, zorder=0)
ax2.set_title("(b)", loc="left", fontweight="bold", color=INK)
med = int(np.median(n_img))
ax2.axhline(med, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=3)
ax2.text(pos[-1] + 0.8, ax2.get_ylim()[1], f"dashed line: median = {med} images per cow", ha="right", va="top", fontsize=6.5, color=INK2)

for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
print("->", a.out + ".{pdf,png}")

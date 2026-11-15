"""BCS distribution in the dataset, by cow and by image (paper figure).

Grouped bars per BCS value: share of cows and share of images (both in %, so they
share one axis). Cows scored more than once (8758, 9008) count once per score,
so the cow shares are over cow-score pairs (67 in Dataset_Fr_BCS_v2).

Usage:
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
rows = []  # (cow, bcs, n_images)
for d in sorted(Path(a.data_dir).iterdir()):
    m = re.fullmatch(r"(\d+)_(\d+(?:\.\d+)?)", d.name)
    if not d.is_dir() or not m:
        continue
    n = sum(1 for p in d.iterdir() if p.suffix.lower() in EXT)
    if n:
        rows.append((m.group(1), float(m.group(2)), n))
n_cows = len({r[0] for r in rows}); n_pairs = len(rows); n_img = sum(r[2] for r in rows)
print(f"{n_cows} cows, {n_pairs} cow-score pairs, {n_img} images")

# Same style as the other manuscript figures (analisis_paper.py, plot_curva.py, roc.py)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
AZUL, NARANJA = "#0072B2", "#D55E00"

levels = np.round(np.arange(2.0, 5.0001, 0.25), 2)
cows = Counter(r[1] for r in rows)
imgs = Counter()
for _, b, n in rows: imgs[b] += n
pc = np.array([100 * cows.get(l, 0) / n_pairs for l in levels])
pi = np.array([100 * imgs.get(l, 0) / n_img for l in levels])

fig, ax = plt.subplots(figsize=(3.5, 2.2))
x = np.arange(len(levels)); w = 0.38
ax.bar(x - w / 2, pc, w, color=AZUL, edgecolor="white", linewidth=0.4, label=f"Cows ($n$ = {n_cows})", zorder=2)
ax.bar(x + w / 2, pi, w, color=NARANJA, edgecolor="white", linewidth=0.4, label=f"Images ($n$ = {n_img})", zorder=2)
ax.set_xticks(x); ax.set_xticklabels([f"{l:.2f}" for l in levels], rotation=45, ha="right", rotation_mode="anchor")
ax.set_xlim(-0.6, len(levels) - 0.4)
ax.set_ylim(0, np.ceil(max(pc.max(), pi.max()) / 5) * 5 + 2)
ax.set_xlabel("BCS"); ax.set_ylabel("Share of dataset (%)")
ax.yaxis.grid(True, color="#e0e0e0", lw=0.5, zorder=0); ax.set_axisbelow(True)
ax.tick_params(length=2.5, width=0.6)
for s in ("left", "bottom"): ax.spines[s].set_linewidth(0.6)
ax.legend(frameon=False, loc="upper left", handlelength=1.2, handleheight=0.8)
fig.tight_layout(pad=0.3)
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
print("->", a.out + ".{pdf,png}")

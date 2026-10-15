#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: make_bcs_distribution.py
# Descripción:
#   Figura de distribución del dataset: número de imágenes por nivel
#   de BCS, leyendo las subcarpetas <BCS>/ de --images_dir (se ignoran
#   carpetas no numéricas, p.ej. "Discarted"). Mismo estilo "paper de
#   ingeniería" que make_paper_figures.py (300 dpi, PNG + PDF).
#
# Uso:
#   python3 make_bcs_distribution.py --images_dir ../../../Dataset/Images \
#       --out_dir ../../../Dataset/figures
# Dependencias: matplotlib (vía make_paper_figures.py)
# =============================================================

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from make_paper_figures import INK_SECONDARY, SERIES_TRAIN, style_axes


def count_by_bcs(images_dir: Path):
    counts = {}
    for d in images_dir.iterdir():
        try:
            bcs = float(d.name)
        except ValueError:
            continue
        if d.is_dir():
            counts[bcs] = sum(1 for p in d.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return dict(sorted(counts.items()))


def plot_distribution(counts, out_dir: Path, title: str):
    levels, n = list(counts), list(counts.values())
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=300)
    style_axes(ax, grid_axis="y")
    x = range(len(levels))
    ax.bar(x, n, width=0.62, color=SERIES_TRAIN, edgecolor="white", linewidth=0.8, zorder=3)
    for xi, v in zip(x, n):
        ax.text(xi, v + max(n) * 0.012, str(v), ha="center", va="bottom", fontsize=8.5, color=INK_SECONDARY)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{b:.2f}" for b in levels], fontsize=9)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylim(0, max(n) * 1.12)
    ax.set_xlabel("Body Condition Score (BCS)")
    ax.set_ylabel("Number of images")
    ax.set_title(f"{title}  (n = {sum(n)})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"bcs_distribution.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Carpeta con subcarpetas <BCS>/")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title", default="BCS distribution in the dataset")
    args = ap.parse_args()
    counts = count_by_bcs(Path(args.images_dir))
    plot_distribution(counts, Path(args.out_dir), args.title)
    print({f"{k:.2f}": v for k, v in counts.items()}, "total", sum(counts.values()))
    print(f"Figura guardada en {Path(args.out_dir).resolve()}/bcs_distribution.png/.pdf")


if __name__ == "__main__":
    main()

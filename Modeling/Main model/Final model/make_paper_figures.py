#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: make_paper_figures.py
# Descripción:
#   Genera figuras estilo "paper de ingeniería" (300 dpi, PNG + PDF
#   vectorial) a partir de los artefactos de una corrida de
#   finalModel.py:
#     - Learning curves (train/val MAE y RMSE por epoch)
#     - Matriz de confusión (BCS discretizado a la rejilla 0.25)
#     - Scatter predicho vs. real, con jitter + código de color por
#       tolerancia + ajuste lineal (estilo "measured vs predicted")
#
#   Estética: caja completa (4 spines), ticks hacia adentro
#   espejados arriba/derecha, tipografía serif (estilo LaTeX),
#   leyenda con marco — la convención clásica de figura de paper/
#   ingeniería, en vez del look de dashboard del skill de dataviz
#   (que es para UI de producto, no para publicación).
#
#   Color: paleta categórica del skill de dataviz para train/val;
#   secuencial azul para magnitud (matriz de confusión); colores de
#   estado (good/critical) para el pass/fail de tolerancia en el
#   scatter — siguen siendo colorblind-safe, solo cambia el "chrome".
#
# Uso:
#   python3 make_paper_figures.py --run_dir <carpeta_con_history.json_y_preds_test.npz> \
#       --out_dir <carpeta_salida>
# Dependencias: numpy, matplotlib, scikit-learn
# =============================================================

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import confusion_matrix, r2_score, mean_absolute_error

# ---- Paleta base (references/palette.md del skill dataviz) ----
INK = "#0b0b0b"            # texto/ejes principales -> negro puro (look de imprenta)
INK_SECONDARY = "#3d3d3b"
GRID = "#d8d7d0"           # gridline hairline, muy recesiva
SURFACE = "#ffffff"        # fondo blanco puro (no el "surface" tibio del dashboard)

SERIES_TRAIN = "#2a78d6"   # azul
SERIES_VAL = "#d03b3b"     # rojo (antes naranja) — probando combo azul/rojo

STATUS_GOOD = "#0ca30c"      # dentro de tolerancia (±0.25 BCS)
STATUS_CRITICAL = "#d03b3b"  # fuera de tolerancia

SEQ_BLUE = ["#eef4fc", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# ---- Estilo "paper de ingeniería": caja completa, ticks adentro,
# serif tipo LaTeX/Computer Modern, sin sombreados de dashboard ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman", "Times"],
    "mathtext.fontset": "cm",
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.linewidth": 1.1,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": True,
    "legend.edgecolor": INK,
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "font.size": 10.5,
})


def style_axes(ax, grid_axis="y"):
    """Caja completa (4 spines), ticks adentro espejados, grid recesiva."""
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(1.1)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid_axis:
        ax.grid(True, axis=grid_axis, which="major", linewidth=0.7, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(which="both", direction="in", top=True, right=True, colors=INK, labelsize=9.5)


def plot_learning_curves(history_path: Path, out_dir: Path, run_label: str):
    with open(history_path) as f:
        h = json.load(f)

    epochs = np.arange(1, len(h["train_mae"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), dpi=300)

    panels = [
        (axes[0], "train_mae", "val_mae", "MAE (BCS)"),
        (axes[1], "train_rmse", "val_rmse", "RMSE (BCS)"),
    ]
    for ax, tkey, vkey, ylabel in panels:
        style_axes(ax)
        if tkey in h and len(h[tkey]) == len(epochs):
            ax.plot(epochs, h[tkey], color=SERIES_TRAIN, linewidth=1.8, label="Train", zorder=3)
        if vkey in h and len(h[vkey]) == len(epochs):
            ax.plot(epochs, h[vkey], color=SERIES_VAL, linewidth=1.8, label="Val", zorder=3)
            best_idx = int(np.argmin(h[vkey]))
            ax.scatter([epochs[best_idx]], [h[vkey][best_idx]], s=42, marker="o",
                       facecolor=SERIES_VAL, edgecolor=INK, linewidth=0.9, zorder=5)
            # Ancla la etiqueta arriba-izquierda del punto (el mínimo suele
            # caer cerca del borde derecho) y le da una tarjeta blanca de
            # fondo para que no se mezcle con las líneas/gridlines de atrás.
            ax.annotate(f"min val = {h[vkey][best_idx]:.3f} (epoch {epochs[best_idx]})",
                        xy=(epochs[best_idx], h[vkey][best_idx]),
                        xytext=(-10, 22), textcoords="offset points",
                        ha="right", fontsize=8.0, color=INK_SECONDARY,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                                  edgecolor=INK_SECONDARY, linewidth=0.6),
                        arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=0.6),
                        zorder=6)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, epochs[-1] + 1)
        ax.set_ylim(bottom=0)

    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle(f"Learning curves — {run_label}", fontsize=12.5, y=1.02)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"learning_curves.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def load_preds(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return d["preds"].astype(float), d["labels"].astype(float)


def bcs_grid(labels: np.ndarray, preds: np.ndarray = None):
    """Rejilla de clases BCS basada en las etiquetas REALES (2.00..5.00 típ.),
    no en las predicciones — una predicción de regresión se puede pasar del
    rango real (p.ej. 5.1) y eso no debe inventar una clase extra vacía."""
    lo = float(np.floor(labels.min() * 4) / 4)
    hi = float(np.ceil(labels.max() * 4) / 4)
    grid = np.round(np.arange(lo, hi + 0.25, 0.25), 2)
    return grid


def snap_to_grid(values: np.ndarray, grid: np.ndarray):
    """Asigna cada valor a la clase más cercana de la rejilla, recortando
    (clip) los valores fuera de rango al extremo más cercano en vez de
    extender la rejilla con clases vacías."""
    clipped = np.clip(values, grid.min(), grid.max())
    idx = np.abs(clipped[:, None] - grid[None, :]).argmin(axis=1)
    return grid[idx], idx


def plot_confusion_matrix(npz_path: Path, out_dir: Path, run_label: str):
    preds, labels = load_preds(npz_path)
    grid = bcs_grid(labels, preds)
    _, true_idx = snap_to_grid(labels, grid)
    _, pred_idx = snap_to_grid(preds, grid)

    cm = confusion_matrix(true_idx, pred_idx, labels=np.arange(len(grid)))
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.clip(row_sums, 1, None)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    fig, ax = plt.subplots(figsize=(6.2, 5.6), dpi=300)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="equal")

    tick_labels = [f"{g:.2f}" for g in grid]
    ax.set_xticks(range(len(grid))); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7.6)
    ax.set_yticks(range(len(grid))); ax.set_yticklabels(tick_labels, fontsize=7.6)
    ax.set_xlabel("Predicted BCS")
    ax.set_ylabel("True BCS")
    ax.set_title(f"Confusion matrix — {run_label}", fontsize=11.5, pad=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                on_diag = i == j
                txt_color = "#ffffff" if cm_norm[i, j] > 0.55 else INK
                weight = "bold" if on_diag else "normal"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=6.6,
                         color=txt_color, fontweight=weight)

    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(INK); spine.set_linewidth(1.1)
    ax.set_xticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", direction="out", top=False, right=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized proportion (per-class recall)", fontsize=8.8)
    cbar.ax.tick_params(labelsize=7.8)
    cbar.outline.set_edgecolor(INK); cbar.outline.set_linewidth(1.0)

    n = cm.sum()
    exact_acc = np.trace(cm) / n if n else 0.0
    off_by_one = (np.trace(cm, offset=-1) + np.trace(cm) + np.trace(cm, offset=1)) / n if n else 0.0
    fig.text(0.5, -0.045,
              f"n = {int(n)}      exact accuracy = {exact_acc:.1%}      ±1-class accuracy (±0.25 BCS) = {off_by_one:.1%}",
              ha="center", fontsize=9, color=INK_SECONDARY)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"confusion_matrix.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_scatter(npz_path: Path, out_dir: Path, run_label: str, tol: float = 0.25, seed: int = 0):
    """
    Predicho vs. real, estilo 'measured vs predicted' de ingeniería:
      - jitter horizontal en 'real' (es discreto, cada 0.25 -> si no,
        se amontonan puntos exactamente encimados y no se lee nada)
      - color por estado: dentro (verde) / fuera (rojo) de tolerancia
      - línea 1:1 ideal (sólida) + banda de tolerancia (líneas punteadas)
      - ajuste lineal por mínimos cuadrados, con ecuación y R²
    """
    preds, labels = load_preds(npz_path)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.055, 0.055, size=labels.shape)
    x_jit = labels + jitter

    within = np.abs(preds - labels) <= tol
    slope, intercept = np.polyfit(labels, preds, 1)

    lo = min(labels.min(), preds.min()) - 0.3
    hi = max(labels.max(), preds.max()) + 0.3

    fig, ax = plt.subplots(figsize=(5.8, 5.6), dpi=300)
    style_axes(ax, grid_axis="both")

    xline = np.array([lo, hi])
    ax.plot(xline, xline, color=INK, linewidth=1.3, zorder=2, label="Ideal (y = x)")
    ax.plot(xline, xline + tol, color=INK, linewidth=0.8, linestyle=(0, (5, 3)), zorder=2)
    ax.plot(xline, xline - tol, color=INK, linewidth=0.8, linestyle=(0, (5, 3)), zorder=2,
            label=f"Tolerance ±{tol:g}")
    ax.plot(xline, slope * xline + intercept, color=SERIES_TRAIN, linewidth=1.6,
            linestyle=(0, (1, 1)), zorder=3, label=f"Fit: y={slope:.2f}x+{intercept:.2f}")

    ax.scatter(x_jit[within], preds[within], s=30, facecolor=STATUS_GOOD, edgecolor=INK,
               linewidth=0.5, alpha=0.85, zorder=4, label=f"Within ±{tol:g} (n={within.sum()})")
    ax.scatter(x_jit[~within], preds[~within], s=30, facecolor=STATUS_CRITICAL, edgecolor=INK,
               linewidth=0.5, alpha=0.85, zorder=4, label=f"Outside ±{tol:g} (n={(~within).sum()})")

    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True BCS (horizontal jitter)")
    ax.set_ylabel("Predicted BCS")
    ax.set_title(f"Predicted vs. true (test) — {run_label}", fontsize=11.5)

    stats_txt = f"MAE = {mae:.3f}\n$R^2$ = {r2:.3f}\nn = {len(labels)}"
    ax.text(0.03, 0.97, stats_txt, transform=ax.transAxes, fontsize=9.5, color=INK, va="top",
            bbox=dict(boxstyle="square,pad=0.4", facecolor=SURFACE, edgecolor=INK, linewidth=0.9))

    ax.legend(loc="lower right", fontsize=7.6, framealpha=1.0)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"scatter_pred_vs_real.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Carpeta con history.json y preds_test.npz")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--run_label", default="ConvNeXt-base + sampler")
    ap.add_argument("--tol", type=float, default=0.25, help="Tolerancia BCS para el scatter (± clases correctas)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(run_dir / "history.json", out_dir, args.run_label)
    plot_confusion_matrix(run_dir / "preds_test.npz", out_dir, args.run_label)
    plot_scatter(run_dir / "preds_test.npz", out_dir, args.run_label, tol=args.tol)

    print(f"Figuras guardadas en {out_dir.resolve()}")


if __name__ == "__main__":
    main()

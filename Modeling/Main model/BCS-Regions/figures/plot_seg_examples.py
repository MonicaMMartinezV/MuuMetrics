# =============================================================
# Nombre del archivo: figures/plot_seg_examples.py
#
# Descripción:
#   Figura de artículo con ejemplos de las anotaciones manuales de regiones
#   anatómicas (Roboflow, cuadros completos del robot) usadas para entrenar el
#   segmentador.
#     (a) varios cuadros: foto recortada a la vaca y sus 6 regiones superpuestas
#     (b) un cuadro descompuesto región por región, para que se vean los traslapes
#   Lee <work>/seg/data.npz que genera `train_segmenter.py prep` (40 cuadros del
#   robot de BCS_masking.sam2 + UNICT; aquí solo se usan los del robot) y no
#   modifica nada.
#
# Uso: python figures/plot_seg_examples.py --data <work>/seg/data.npz [--ids 3 22 29 13] [--detalle 22]
#      -> figures/segmentation_examples.{pdf,png}
# =============================================================
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
from PIL import Image

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]          # mismo orden que segmenter.REGIONS
NOMBRES = ["Loin", "Hips", "Rump", "Tailhead", "Thigh (image left)", "Thigh (image right)"]
COLORES = ["#E69F00", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]   # Okabe-Ito, apto para daltonismo

plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.titlesize": 8, "legend.fontsize": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42})

def recorte(img, masks, margen=0.06, salida=520):
    """Recorte CUADRADO alrededor de la unión de regiones, siempre dentro del cuadro (sin relleno):
    lado = el mayor de alto/ancho de las regiones + margen, limitado al alto del cuadro; se desliza para
    no salirse y se lleva a salida x salida px."""
    ys, xs = np.where(masks.max(0) > 0); h, w = img.shape[:2]
    lado = int(min(max(ys.max() - ys.min(), xs.max() - xs.min()) * (1 + 2 * margen), h, w))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    y0 = int(np.clip(round(cy - lado / 2), 0, h - lado)); x0 = int(np.clip(round(cx - lado / 2), 0, w - lado))
    im = np.asarray(Image.fromarray(img[y0:y0 + lado, x0:x0 + lado]).resize((salida, salida), Image.LANCZOS))
    ms = np.stack([np.asarray(Image.fromarray(m[y0:y0 + lado, x0:x0 + lado] * 255).resize((salida, salida), Image.NEAREST)) > 127
                   for m in masks]).astype(np.uint8)
    return im, ms

def superponer(img, masks, regiones=range(6), alpha=0.38):
    """Relleno semitransparente por región (los traslapes mezclan colores)."""
    out = img.astype(np.float32) / 255.0
    for r in regiones:
        m = masks[r] > 0
        out[m] = (1 - alpha) * out[m] + alpha * np.array(to_rgb(COLORES[r]))
    return np.clip(out, 0, 1)

def contornos(ax, masks, regiones=range(6), lw=1.0):
    for r in regiones:
        if masks[r].any(): ax.contour(masks[r] > 0, levels=[0.5], colors=[COLORES[r]], linewidths=lw)

def limpiar(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("0.4")

def main(a):
    z = np.load(a.data); robot = np.where(z["src"] == "robot")[0]
    imgs, masks = z["imgs"], z["masks"]
    ids = [robot[i] for i in a.ids]; det = robot[a.detalle]
    n = len(ids)
    H = 4.9; fig = plt.figure(figsize=(7.0, H))
    L, R, g = 0.075, 0.99, 0.035                                    # márgenes y separación (fracción del panel)
    pa = (R - L) * 7.0 / (n + (n - 1) * g); pb = (R - L) * 7.0 / (6 + 5 * g)   # lado de panel (pulgadas) en (a) y (b)
    t_a = 0.99; b_a = t_a - (2 * pa + g * pa) / H
    t_b = b_a - 0.30 / H; b_b = t_b - pb / H
    top = fig.add_gridspec(2, n, hspace=g, wspace=g, left=L, right=R, top=t_a, bottom=b_a)
    bot = fig.add_gridspec(1, 6, wspace=g, left=L, right=R, top=t_b, bottom=b_b)
    gs = None
    # (a) ejemplos: foto arriba, anotación abajo
    for j, i in enumerate(ids):
        im, ms = recorte(imgs[i], masks[i])
        ax = fig.add_subplot(top[0, j]); ax.imshow(im); limpiar(ax)
        ax2 = fig.add_subplot(top[1, j]); ax2.imshow(superponer(im, ms)); contornos(ax2, ms); limpiar(ax2)
        if j == 0:
            ax.set_ylabel("Robot frame", labelpad=3); ax2.set_ylabel("Manual annotation", labelpad=3)
            ax.text(-0.17, 1.0, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="right")
    # (b) un cuadro, región por región
    im, ms = recorte(imgs[det], masks[det])
    gris = np.repeat((im.astype(np.float32) @ [0.299, 0.587, 0.114])[..., None], 3, 2).astype(np.uint8)
    for r in range(6):
        ax = fig.add_subplot(bot[0, r]); ax.imshow(superponer(gris, ms, [r], alpha=0.55)); contornos(ax, ms, [r], lw=0.9)
        otras = [k for k in range(6) if k != r]
        for k in otras:
            if ms[k].any(): ax.contour(ms[k] > 0, levels=[0.5], colors=["0.85"], linewidths=0.35, linestyles="--")
        ax.set_title(NOMBRES[r], pad=2); limpiar(ax)
        if r == 0: ax.text(-0.30, 1.0, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="right")
    fig.legend(handles=[Patch(facecolor=c, edgecolor=c, alpha=0.75, label=nm) for c, nm in zip(COLORES, NOMBRES)],
               loc="lower center", ncol=6, frameon=False, handlelength=1.2, columnspacing=1.1, bbox_to_anchor=(0.53, -0.005))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"-> {out.with_suffix('.pdf')} / .png | ejemplos {[str(z['names'][i]) for i in ids]} | detalle {z['names'][det]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="<work>/seg/data.npz de train_segmenter.py prep")
    ap.add_argument("--ids", type=int, nargs="+", default=[3, 22, 29, 13], help="índices entre los 40 cuadros del robot")
    ap.add_argument("--detalle", type=int, default=22, help="cuadro para el panel (b)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "segmentation_examples"))
    main(ap.parse_args())

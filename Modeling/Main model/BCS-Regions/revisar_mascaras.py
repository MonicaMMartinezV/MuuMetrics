# =============================================================
# Nombre del archivo: revisar_mascaras.py
#
# Descripción:
#   Lámina por vaca con las máscaras del segmentador (caché <work>/masks)
#   dibujadas sobre las fotos, para las vacas×BCS peor predichas en
#   <work>/preds_loco.csv. Cada región: relleno tenue + contorno (las
#   regiones se traslapan, por eso no se pinta una sola etiqueta por píxel).
#
# Uso (en el host, desde esta carpeta):
#   python3 revisar_mascaras.py [--work work] [--data ../../../Dataset/Dataset_Fr_BCS] [--n 8] [--fotos 6]
#   -> <work>/revision_mascaras/NN_<vaca>_<bcs>.png
# =============================================================
import argparse, csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
NOMBRES = ["lomo", "cadera", "grupa", "base de la cola", "muslo izq.", "muslo der."]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]

def spread(lst, k):
    return lst if len(lst) <= k else [lst[round(i * (len(lst) - 1) / (k - 1))] for i in range(k)]

def main(a):
    work, data = Path(a.work), Path(a.data)
    out = work / "revision_mascaras"; out.mkdir(exist_ok=True)
    g = defaultdict(list)
    for r in csv.DictReader(open(work / "preds_loco.csv")):
        g[(r["vaca"], float(r["bcs_real"]))].append((r["foto"], float(r["bcs_pred"])))
    peores = sorted(g.items(), key=lambda kv: -abs(np.mean([p for _, p in kv[1]]) - kv[0][1]))[:a.n]
    for k, ((vaca, bcs), fotos) in enumerate(peores, 1):
        fotos = spread(sorted(fotos), a.fotos); n = len(fotos); cols = min(n, 3); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows + 0.9), squeeze=False, facecolor="white")
        for ax, (name, pred) in zip(axes.flat, fotos):
            img = np.asarray(Image.open(data / name).convert("RGB")); h, w = img.shape[:2]
            m = np.load(work / "masks" / Path(name).with_suffix(".npz"))["masks"].astype(np.float32) / 255.0
            ax.imshow(img)
            for i, c in enumerate(COLORS):
                mi = np.asarray(Image.fromarray(m[i]).resize((w, h), Image.BILINEAR))
                rgba = np.zeros((h, w, 4)); rgba[..., :3] = matplotlib.colors.to_rgb(c); rgba[..., 3] = (mi > 0.5) * 0.18
                ax.imshow(rgba)
                if mi.max() > 0.5: ax.contour(mi, levels=[0.5], colors=[c], linewidths=1.8)
            ax.set_title(f"{Path(name).name.split('__')[-1][:10]} · predicho {pred:.2f}", fontsize=9, color="#52514e")
        for ax in axes.flat: ax.axis("off")
        media = np.mean([p for _, p in g[(vaca, bcs)]])
        fig.suptitle(f"Vaca {vaca} · BCS real {bcs:.2f} · predicho {media:.2f} (promedio de {len(g[(vaca, bcs)])} fotos)",
                     x=0.02, ha="left", fontsize=13, fontweight="bold")
        fig.legend(handles=[Patch(facecolor=c, edgecolor=c, label=t) for c, t in zip(COLORS, NOMBRES)],
                   loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        dst = out / f"{k:02d}_{vaca}_{bcs:g}.png"; fig.savefig(dst, dpi=110); plt.close(fig); print(dst)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="work")
    ap.add_argument("--data", default="../../../../MuuMetrics_Adri/Dataset/Dataset_Fr_BCS")
    ap.add_argument("--n", type=int, default=8, help="cuántas vacas×BCS (las peores)")
    ap.add_argument("--fotos", type=int, default=6, help="máximo de fotos por vaca")
    main(ap.parse_args())

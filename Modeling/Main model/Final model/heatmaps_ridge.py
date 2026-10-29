# =============================================================
# Nombre del archivo: heatmaps_ridge.py
#
# Descripción:
#   Mapas de relevancia del modelo DINOv2-L por región + ridge.
#   El modelo es lineal sobre el promedio de patches de cada región:
#     pred = b + sum_r beta_r · (mean_{p in r} f_p - mu_r) / sd_r + absent
#   así que la predicción se reparte EXACTAMENTE entre los patches:
#     c_p = sum_r a_rp · beta_r · (f_p - mu_r) / sd_r,  a_rp = peso del patch en r
#   y  pred = b + sum_p c_p + término de regiones ausentes.
#   c_p > 0: el patch empuja el BCS por arriba de la vaca promedio del
#   train; c_p < 0: por abajo. |c_p| = dónde mira el modelo.
#   Cada foto se explica con el ridge entrenado SIN su vaca (leave-one-
#   cow-out, alpha por GroupKFold(4) como en robust_cv.py).
#
# Salidas (<out_dir>): heatmap_<foto>.png, panel_heatmaps.png,
#   region_importance.png, region_contrib.csv
# =============================================================
import argparse, csv
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import timm
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from filmModel import load_records
from backbones_frozen import mask_path, MEAN, STD

REGIONS = ["lomo", "cadera", "grupa", "base de cola", "muslo izq.", "muslo der."]
ALPHAS = (1, 10, 100, 1e3, 1e4, 1e5)
SIZE, PATCH, C = 378, 14, 1024

def loco_ridge(X, y, cows, cow):
    """Ridge sin la vaca `cow` (misma selección de alpha que robust_cv.py)."""
    tr = cows != cow
    sc = StandardScaler().fit(X[tr]); Xtr = sc.transform(X[tr]); ytr, gtr = y[tr], cows[tr]
    best, best_err = None, np.inf
    for a in ALPHAS:
        err = [np.abs(Ridge(alpha=a).fit(Xtr[i], ytr[i]).predict(Xtr[v]) - ytr[v]) for i, v in GroupKFold(4).split(Xtr, ytr, gtr)]
        e = float(np.mean(np.concatenate(err)))
        if e < best_err: best, best_err = a, e
    m = Ridge(alpha=best).fit(Xtr, ytr)
    return m.coef_ / sc.scale_, sc.mean_, float(m.intercept_)                   # intercept = predicción de la vaca promedio

def region_contrib(x, B, mu):
    """Contribución de cada región (y de los indicadores de ausencia) a pred - pred(vaca promedio)."""
    d = (x - mu) * B
    return d[:6 * C].reshape(6, C).sum(1), d[6 * C:].sum()

@torch.no_grad()
def patch_map(model, path, masks_dir, B, mu, dev):
    img = Image.open(path).convert("RGB")
    x = np.asarray(img.resize((SIZE, SIZE), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        t = model.forward_features(x)
    f = t[0, model.num_prefix_tokens:].float().cpu().numpy()                     # [729, C]
    m = np.load(mask_path(path, masks_dir))["masks"].astype(np.float32) / 255.0
    m = F.interpolate(torch.from_numpy((m > 0.5).astype(np.float32))[None], size=(SIZE, SIZE), mode="nearest")
    w = F.adaptive_avg_pool2d(m, (SIZE // PATCH,) * 2)[0].reshape(6, -1).numpy()  # [6, 729]
    tot = w.sum(1, keepdims=True); a = np.where(tot >= 0.5, w / np.maximum(tot, 1e-6), 0)
    Br, mur = B[:6 * C].reshape(6, C), mu[:6 * C].reshape(6, C)
    c = np.einsum("rp,rp->p", a, np.einsum("rc,pc->rp", Br, f) - (Br * mur).sum(1, keepdims=True))
    reg = (a @ f).reshape(-1)                                                     # para verificar contra el caché
    return img, m[0].numpy(), c.reshape(SIZE // PATCH, -1), reg

def upsample(cmap_, hw):
    t = torch.from_numpy(cmap_)[None, None].float()
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()

def draw(axes, img, masks, c, title, lim):
    W, H = img.size; gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    up = upsample(c, (H, W)); mk = F.interpolate(torch.from_numpy(masks)[None], size=(H, W), mode="nearest")[0].numpy()
    axes[0].imshow(img); axes[0].set_title(title, fontsize=10, loc="left")
    for r, col in enumerate(plt.cm.tab10(range(6))):
        axes[0].contour(mk[r], levels=[0.5], colors=[col], linewidths=1.2)
        axes[0].plot([], [], color=col, label=REGIONS[r])
    axes[0].legend(loc="lower left", fontsize=7, framealpha=0.8, handlelength=1.2)
    rel = np.abs(up); axes[1].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[1].imshow(rel, cmap="inferno", alpha=np.clip(rel / rel.max(), 0, 1) * 0.85, vmin=0, vmax=rel.max())
    axes[1].set_title("Relevancia |contribución|", fontsize=10, loc="left")
    axes[2].imshow(gray, cmap="gray", vmin=0, vmax=1)
    axes[2].imshow(up, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), alpha=np.clip(np.abs(up) / lim, 0, 1) * 0.9)
    axes[2].set_title("Rojo sube el BCS · azul lo baja", fontsize=10, loc="left")
    for ax in axes: ax.axis("off")

def main(a):
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    names = [Path(r["path"]).name for r in records]
    f = np.load(a.features); X = np.hstack([f["reg"], f["absent"]])

    # Ridge LOCO por vaca; contribución por región de cada foto
    fits, R, P = {}, np.zeros((len(y), 6)), np.zeros(len(y))
    for cow in np.unique(cows):
        B, mu, base = loco_ridge(X, y, cows, cow)
        fits[cow] = (B, mu, base)
        for i in np.where(cows == cow)[0]:
            R[i], ab = region_contrib(X[i], B, mu); P[i] = base + R[i].sum() + ab
    print(f"MAE LOCO = {np.abs(P - y).mean():.3f}")
    with open(out / "region_contrib.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred", *REGIONS])
        w.writerows([[n, c, yy, round(p, 3), *np.round(r, 3)] for n, c, yy, p, r in zip(names, cows, y, P, R)])

    # Importancia por región en todo el dataset
    fig, ax = plt.subplots(figsize=(7, 3.2))
    imp = np.abs(R).mean(0); o = np.argsort(imp)
    ax.barh([REGIONS[k] for k in o], imp[o], color="#2E6F57")
    ax.set_xlabel("|contribución| media al BCS (puntos)"); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Peso de cada región en la predicción ({len(y)} fotos, {len(np.unique(cows))} vacas)", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(out / "region_importance.png", dpi=200); plt.close(fig)

    # Fotos a mostrar: por cada nivel de BCS, la foto mejor predicha de una vaca distinta
    err = np.abs(P - y); chosen, used = [], set()
    for lvl in a.levels:
        idx = [i for i in np.argsort(err) if abs(y[i] - lvl) < 0.13 and cows[i] not in used]
        if idx: chosen.append(idx[0]); used.add(cows[idx[0]])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=SIZE).to(dev).eval()
    maps = []
    for i in chosen:
        B, mu, base = fits[cows[i]]
        img, masks, c, reg = patch_map(model, records[i]["path"], Path(a.masks_dir), B, mu, dev)
        maps.append((i, img, masks, c))
        print(f"{names[i]}: vaca {cows[i]} real {y[i]} pred {P[i]:.2f} | suma de patches {base + c.sum() + region_contrib(X[i], B, mu)[1]:.2f}"
              f" | dif. features vs caché {np.abs(reg - f['reg'][i]).max():.3f}")
    lim = np.percentile(np.abs(np.concatenate([m[3].ravel() for m in maps])), 99.5)
    for i, img, masks, c in maps:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.4 * img.size[1] / img.size[0] + 0.6))
        draw(axes, img, masks, c, f"Vaca {cows[i]} · BCS real {y[i]:.2f} · predicho {P[i]:.2f}", lim)
        fig.tight_layout(); fig.savefig(out / f"heatmap_{Path(names[i]).stem}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(maps), 3, figsize=(15, 5.4 * len(maps) * maps[0][1].size[1] / maps[0][1].size[0] + 0.6))
    for row, (i, img, masks, c) in zip(np.atleast_2d(axes), maps):
        draw(row, img, masks, c, f"Vaca {cows[i]} · real {y[i]:.2f} · pred. {P[i]:.2f}", lim)
    fig.tight_layout(); fig.savefig(out / "panel_heatmaps.png", dpi=130); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv"); ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--masks_dir", default="Dataset_Fr_BCS_masks")
    ap.add_argument("--features", default="outputs_backbones_cv/features_dinov2L.npz")
    ap.add_argument("--out_dir", default="outputs_backbones_cv/heatmaps")
    ap.add_argument("--levels", type=float, nargs="+", default=[2.5, 3.5, 4.0, 4.5, 5.0])
    main(ap.parse_args())

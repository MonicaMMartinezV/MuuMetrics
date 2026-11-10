# =============================================================
# Nombre del archivo: viz_masks.py
#
# Descripción:
#   Dibuja las 6 regiones del segmentador (entrenado con Roboflow)
#   sobre cada foto. Por foto genera un PNG con:
#     [foto original | regiones superpuestas | 6 mapas de probabilidad]
#
#   - Con máscaras ya calculadas (.npz del paso 'masks'): solo numpy + PIL,
#     corre en el host sin GPU.
#   - Con --segment: corre el segmentador sobre fotos nuevas (necesita
#     torch/timm y los pesos -> usar dentro del contenedor).
#
# Uso (host):
#   python3 viz_masks.py --images <carpeta de fotos> --masks <carpeta de npz> --out <carpeta>
# Uso (Docker, fotos nuevas):
#   docker compose run --rm --entrypoint python bcs viz_masks.py --segment \
#       --images /data/<carpeta> --out /work/viz_nuevas
# =============================================================
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REGIONS = ["loin", "hips", "rump", "tailhead", "thigh_L", "thigh_R"]
NOMBRES = ["lomo", "cadera", "grupa", "base cola", "muslo izq", "muslo der"]
COLORS = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180]], float)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
H = 360  # alto de cada panel

def resize(a: np.ndarray, size) -> np.ndarray:
    return np.asarray(Image.fromarray(a).resize(size, Image.BILINEAR))

def edges(m: np.ndarray) -> np.ndarray:
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] != m[:-1, :]; e[:, 1:] |= m[:, 1:] != m[:, :-1]
    return e & m

def render(img: np.ndarray, probs: np.ndarray, title: str) -> Image.Image:
    """img uint8 [h,w,3], probs float [6,h',w'] en [0,1] -> panel PNG."""
    h, w = img.shape[:2]; W = int(round(w * H / h))
    base = resize(img, (W, H)).astype(float)
    p = np.stack([resize((c * 255).astype(np.uint8), (W, H)) for c in probs]).astype(float) / 255
    over = base.copy()
    for k in range(6):  # las regiones se traslapan: se mezclan una sobre otra
        m = p[k] > 0.5
        over[m] = 0.55 * over[m] + 0.45 * COLORS[k]
    for k in range(6):
        over[edges(p[k] > 0.5)] = COLORS[k]
    small_h = H // 2; small_w = int(round(W / 2))
    heat = []
    for k in range(6):
        g = resize(img, (small_w, small_h)).astype(float).mean(2, keepdims=True) * 0.5
        pk = resize((p[k] * 255).astype(np.uint8), (small_w, small_h)).astype(float)[..., None] / 255
        heat.append((g * (1 - pk) + COLORS[k] * pk).clip(0, 255).astype(np.uint8))
    grid = np.concatenate([np.concatenate(heat[0:3], 1), np.concatenate(heat[3:6], 1)], 0)
    panel = np.full((H + 60, W * 2 + grid.shape[1] + 20, 3), 255, np.uint8)
    panel[30:30 + H, 0:W] = base.astype(np.uint8)
    panel[30:30 + H, W + 10:2 * W + 10] = over.clip(0, 255).astype(np.uint8)
    panel[30:30 + H, 2 * W + 20:2 * W + 20 + grid.shape[1]] = grid
    im = Image.fromarray(panel); d = ImageDraw.Draw(im); f = ImageFont.load_default()
    d.text((4, 8), title, fill=(0, 0, 0), font=f)
    d.text((W + 14, 8), "regiones (prob > 0.5)", fill=(0, 0, 0), font=f)
    d.text((2 * W + 24, 8), "probabilidad por region", fill=(0, 0, 0), font=f)
    for k in range(6):  # nombres sobre cada mapa + leyenda abajo
        r, c = divmod(k, 3)
        d.text((2 * W + 24 + c * small_w, 32 + r * small_h), NOMBRES[k], fill=(255, 255, 255), font=f)
        x = 4 + k * 110
        d.rectangle([x, H + 38, x + 12, H + 50], fill=tuple(int(v) for v in COLORS[k]))
        area = (p[k] > 0.5).mean() * 100
        d.text((x + 16, H + 38), f"{NOMBRES[k]} {area:.0f}%", fill=(0, 0, 0), font=f)
    return im

def main():
    ap = argparse.ArgumentParser(description="Visualiza las regiones del segmentador sobre las fotos")
    ap.add_argument("--images", required=True, help="carpeta de fotos (recursivo) o una foto")
    ap.add_argument("--masks", help="carpeta de .npz con la misma estructura que --images")
    ap.add_argument("--segment", action="store_true", help="calcular las máscaras con el segmentador (requiere torch)")
    ap.add_argument("--seg_weights", default="/weights/final_all40.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=0, help="máximo de fotos (0 = todas)")
    a = ap.parse_args()
    if not a.masks and not a.segment: ap.error("usa --masks <carpeta> o --segment")
    src = Path(a.images)
    paths = sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    if a.max: paths = paths[:a.max]
    if a.segment:
        import torch
        from segmenter import load_segmenter, segment
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); net = load_segmenter(a.seg_weights, dev)
    out = Path(a.out); n = 0
    for p in paths:
        rel = p.relative_to(src) if src.is_dir() else Path(p.name)
        img = np.asarray(Image.open(p).convert("RGB"))
        if a.segment:
            probs = segment(net, img, dev)
        else:
            f = Path(a.masks) / rel.with_suffix(".npz")
            if not f.exists(): print(f"  sin máscara: {f}"); continue
            probs = np.load(f)["masks"].astype(np.float32) / 255
        dst = out / rel.with_suffix(".png"); dst.parent.mkdir(parents=True, exist_ok=True)
        render(img, probs, str(rel)).save(dst); n += 1
    print(f"[viz] {n} imágenes -> {out}")

if __name__ == "__main__":
    main()

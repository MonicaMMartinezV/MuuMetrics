# =============================================================
# Nombre del archivo: luz/normalizacion.py
#
# Descripción:
#   Experimento: ¿ayuda normalizar la luminosidad antes de DINOv2?
#   Las máscaras (caché <work>/masks) no cambian; solo se recalculan los
#   descriptores por región con cada variante y se evalúa igual que
#   bcs_regions.py evaluate (LOCO y 5 folds por vaca x semillas).
#     ninguna       referencia (copia <work>/features.npz)
#     gamma         gamma en L (Lab) para que la media de L dentro de la vaca sea 0.5
#     gris          balance de blancos gray-world
#     clahe1/2/4    CLAHE en L (Lab), malla 8x8, clipLimit 1/2/4
#     gris_clahe2   gray-world y luego CLAHE 2
#   Salidas: <work>/luz/<variante>/{features.npz,metrics.csv,preds_*.csv},
#            <work>/luz/resumen.csv y <work>/luz/ejemplos.jpg
#
# Uso (necesita opencv; se instala una vez en /work/.pylib):
#   docker compose run --rm --entrypoint sh -e PYTHONPATH=/work/.pylib bcs -c \
#     "pip install -q --target /work/.pylib opencv-python-headless==4.10.0.84 && python luz/normalizacion.py"
# =============================================================
import argparse, csv, shutil, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bcs_regions as br
from data import load_records

def on_L(img, f):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB); lab[..., 0] = f(lab[..., 0]); return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def clahe(clip):
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return lambda img, cow: on_L(img, op.apply)

def gris(img, cow):
    x = img.astype(np.float32); mean = x.reshape(-1, 3).mean(0)
    return np.clip(x * (mean.mean() / np.maximum(mean, 1)), 0, 255).astype(np.uint8)

def gamma(img, cow):
    def f(L):
        l = L.astype(np.float32) / 255; mu = np.clip(l[cow].mean() if cow.any() else l.mean(), 0.02, 0.98)
        return np.clip((l ** (np.log(0.5) / np.log(mu))) * 255, 0, 255).astype(np.uint8)
    return on_L(img, f)

VARIANTS = {"ninguna": None, "gamma": gamma, "gris": gris, "clahe1": clahe(1), "clahe2": clahe(2), "clahe4": clahe(4),
            "gris_clahe2": lambda img, cow: clahe(2)(gris(img, cow), cow)}

def cow_mask(masks, shape):
    u = (masks.max(0) > 127).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(u).resize((shape[1], shape[0]), Image.NEAREST)) > 0

def main(a):
    work = Path(a.work_dir); out = work / "luz"; recs = load_records(a.data_dir); dev = br.device(); bb = None
    for v, fn in VARIANTS.items():
        wv = out / v; wv.mkdir(parents=True, exist_ok=True)
        if not (wv / "features.npz").exists():
            if fn is None: shutil.copy(work / "features.npz", wv / "features.npz")
            else:
                bb = bb or br.load_backbone(dev); X = []
                for r in recs:
                    mz = np.load(br.mask_file(work, r))["masks"]; img = br.read_rgb(r["path"])
                    X.append(br.describe(bb, fn(img, cow_mask(mz, img.shape)), mz.astype(np.float32) / 255.0, dev))
                np.savez_compressed(wv / "features.npz", X=np.stack(X).astype(np.float32), y=np.array([r["bcs"] for r in recs]),
                                    vaca=np.array([r["vaca"] for r in recs]), name=np.array([r["name"] for r in recs]))
        print(f"\n===== {v} ====="); br.cmd_evaluate(argparse.Namespace(work_dir=str(wv), seeds=a.seeds))
    rows = []
    for v in VARIANTS:
        for r in csv.DictReader(open(out / v / "metrics.csv")): rows.append({"variante": v, **r})
    with open(out / "resumen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\nvariante      esquema   MAE    ±0.5   MAE por vaca")
    for r in rows: print(f"{r['variante']:13s} {r['esquema']:8s} {float(r['mae']):.3f}  {float(r['acc_05']):.0%}   {float(r['mae_vaca']):.3f}")

    # Lámina de ejemplo: la foto más oscura, una mediana y la más clara (brillo dentro de la vaca)
    lum = []
    for r in recs:
        mz = np.load(br.mask_file(work, r))["masks"]; img = br.read_rgb(r["path"]); c = cow_mask(mz, img.shape)
        lum.append(cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[..., 0][c].mean() if c.any() else 0)
    pick = [recs[i] for i in np.argsort(lum)[[0, len(lum) // 2, -1]]]; H = 220; tiles = []
    for r in pick:
        mz = np.load(br.mask_file(work, r))["masks"]; img = br.read_rgb(r["path"]); c = cow_mask(mz, img.shape); row = []
        for v, fn in VARIANTS.items():
            im = Image.fromarray(img if fn is None else fn(img, c)); im = im.resize((round(im.width * H / im.height), H)); row.append(im)
        tiles.append(row)
    W = max(sum(im.width for im in row) for row in tiles); sheet = Image.new("RGB", (W + 4 * len(VARIANTS), (H + 4) * len(tiles)), "white")
    for j, row in enumerate(tiles):
        x = 0
        for im in row: sheet.paste(im, (x, j * (H + 4))); x += im.width + 4
    sheet.save(out / "ejemplos.jpg", quality=88); print(f"columnas: {list(VARIANTS)} -> {out / 'ejemplos.jpg'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS")
    ap.add_argument("--work_dir", default="/work")
    ap.add_argument("--seeds", type=int, default=10)
    main(ap.parse_args())

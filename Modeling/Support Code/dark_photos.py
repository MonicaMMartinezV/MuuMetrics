#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: dark_photos.py
# Descripción: Mide el brillo de cada foto de ByID (<ids_dir>/<vaca>/img.jpg)
#   para detectar las fotos demasiado oscuras para calificar BCS.
#   - brillo = media del canal L (0-255) del recorte; contraste = desviación.
#   - Escribe brillo_fotos.csv (una fila por foto) y una hoja con las
#     --n_sheet fotos más oscuras, para elegir el umbral a ojo.
#   - Con --cv_dir, cruza con las predicciones de test del CV
#     (fold*/preds_test.npz) y reporta el MAE por rango de brillo.
#
# Uso (numpy está en la imagen muumetrics-convnext):
#   python dark_photos.py --ids_dir Dataset/ByID --out_dir Dataset/dark_photos \
#       --cv_dir Dataset/outputs_byid_cv
# =============================================================

import argparse, csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from review_bcs_jumps import fit, load_font

BINS = [0, 30, 45, 60, 80, 256]
CELL_W, CELL_H, LABEL_H, COLS = 220, 260, 24, 8


def measure(ids_dir: Path):
    rows = []
    for p in sorted(ids_dir.glob("*/*.jpg")):
        if not p.parent.name.isdigit():
            continue
        im = Image.open(p).convert("L")
        im.thumbnail((256, 256))
        a = np.asarray(im, np.float32)
        rows.append({"vaca": p.parent.name, "img": p.name.replace(" - Copy", ""), "path": p,
                     "brillo": round(float(a.mean()), 1), "contraste": round(float(a.std()), 1),
                     "camara": p.name[20:24]})
    return rows


def load_cv_errors(cv_dir: Path):
    err = {}
    for f in sorted(cv_dir.glob("fold*/preds_test.npz")):
        d = np.load(f, allow_pickle=True)
        for path, pred, lab in zip(d["paths"], d["preds"], d["labels"]):
            err[Path(str(path)).name.replace(" - Copy", "")] = (float(pred), float(lab))
    return err


def dark_sheet(rows, n, out_path):
    rows = sorted(rows, key=lambda r: r["brillo"])[:n]
    nrows = (len(rows) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * (CELL_W + 6) + 6, nrows * (CELL_H + LABEL_H + 6) + 6), "white")
    draw, font = ImageDraw.Draw(sheet), load_font(14)
    for k, r in enumerate(rows):
        x, y = 6 + (k % COLS) * (CELL_W + 6), 6 + (k // COLS) * (CELL_H + LABEL_H + 6)
        sheet.paste(fit(Image.open(r["path"]).convert("RGB"), CELL_W, CELL_H), (x, y))
        draw.text((x + 2, y + CELL_H + 3), f"brillo {r['brillo']:.0f}  vaca {r['vaca']}  {r['camara']}",
                  fill="black", font=font)
    sheet.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cv_dir", default=None)
    ap.add_argument("--n_sheet", type=int, default=64)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = measure(Path(args.ids_dir))
    err = load_cv_errors(Path(args.cv_dir)) if args.cv_dir else {}
    for r in rows:
        pred, lab = err.get(r["img"], (None, None))
        r["bcs"], r["pred_cv"] = lab, pred
        r["error_abs"] = round(abs(pred - lab), 3) if pred is not None else None

    with open(out_dir / "brillo_fotos.csv", "w", newline="") as f:
        fields = ["vaca", "img", "camara", "brillo", "contraste", "bcs", "pred_cv", "error_abs"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["brillo"]))
    dark_sheet(rows, args.n_sheet, out_dir / "mas_oscuras.jpg")

    b = np.array([r["brillo"] for r in rows])
    print(f"{len(rows)} fotos | brillo: mediana {np.median(b):.0f}, p10 {np.percentile(b, 10):.0f}, "
          f"p90 {np.percentile(b, 90):.0f}")
    for cam in sorted({r["camara"] for r in rows}):
        bc = [r["brillo"] for r in rows if r["camara"] == cam]
        print(f"  {cam}: {len(bc)} fotos, brillo mediano {np.median(bc):.0f}")
    print(f"\n{'brillo':>9} {'fotos':>6} {'con pred':>9} {'MAE CV':>7} {'BCS medio':>10}")
    for lo, hi in zip(BINS, BINS[1:]):
        sel = [r for r in rows if lo <= r["brillo"] < hi]
        e = [r["error_abs"] for r in sel if r["error_abs"] is not None]
        bcs = [r["bcs"] for r in sel if r["bcs"] is not None]
        print(f"{lo:>3}-{hi:<5} {len(sel):6d} {len(e):9d} {np.mean(e) if e else float('nan'):7.3f} "
              f"{np.mean(bcs) if bcs else float('nan'):10.2f}")
    print(f"\nCSV: {out_dir / 'brillo_fotos.csv'} | hoja: {out_dir / 'mas_oscuras.jpg'}")


if __name__ == "__main__":
    main()

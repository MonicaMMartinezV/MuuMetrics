# =============================================================
# Nombre del archivo: crop_dataset.py
# Descripción: Recorta un dataset completo de fotos sin recortar
#   (carpetas por BCS) con CowScanner.pt, conservando la estructura
#   <BCS>/<imagen>.jpg para entrenar directamente.
#   - Usa la caja de mayor confianza (igual que CropCow.py).
#   - --margin agrega un margen relativo alrededor de la caja.
#   - Ignora carpetas cuyo nombre no es un número (p.ej. "Discarted").
#   - Escribe crops_log.csv con la caja, confianza y las fotos sin
#     detección (que no se copian).
#
# Uso (en la imagen muumetrics-tracking, que trae ultralytics):
#   python crop_dataset.py --src /data/Images --dst /data/Images_crops \
#       --weights CowScanner.pt --margin 0.05
#
# Dependencias: ultralytics, opencv
# =============================================================

import argparse, csv
from pathlib import Path

import cv2
from ultralytics import YOLO


def is_label(name: str) -> bool:
    try:
        float(name)
        return True
    except ValueError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Carpeta con subcarpetas <BCS>/ de fotos sin recortar")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--weights", default="CowScanner.pt")
    ap.add_argument("--margin", type=float, default=0.05, help="Margen relativo al tamaño de la caja")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(args.weights)
    src, dst = Path(args.src), Path(args.dst)
    log = []
    folders = sorted(p for p in src.iterdir() if p.is_dir() and is_label(p.name))
    for folder in folders:
        (dst / folder.name).mkdir(parents=True, exist_ok=True)
        for img_path in sorted(folder.glob("*.jpg")):
            image = cv2.imread(str(img_path))
            res = model(image, conf=args.conf, verbose=False)[0]
            row = {"label": folder.name, "image": img_path.name, "conf": "", "x1": "", "y1": "", "x2": "", "y2": ""}
            if len(res.boxes) == 0:
                log.append(row)
                continue
            best = int(res.boxes.conf.argmax())
            x1, y1, x2, y2 = res.boxes.xyxy[best].tolist()
            mx, my = (x2 - x1) * args.margin, (y2 - y1) * args.margin
            H, W = image.shape[:2]
            x1, y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            x2, y2 = min(W, int(x2 + mx)), min(H, int(y2 + my))
            cv2.imwrite(str(dst / folder.name / img_path.name), image[y1:y2, x1:x2])
            row.update(conf=round(float(res.boxes.conf[best]), 4), x1=x1, y1=y1, x2=x2, y2=y2)
            log.append(row)

    with open(dst / "crops_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader(); w.writerows(log)
    missing = [r for r in log if r["conf"] == ""]
    print(f"Fotos: {len(log)} | recortadas: {len(log) - len(missing)} | sin detección: {len(missing)}")
    for r in missing[:10]:
        print("  sin detección:", r["label"], r["image"])


if __name__ == "__main__":
    main()

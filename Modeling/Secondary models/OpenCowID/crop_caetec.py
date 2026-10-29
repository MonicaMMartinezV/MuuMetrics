# =============================================================
# Nombre del archivo: crop_caetec.py
# Descripción: Recorta la vaca principal de las fotos de confirmación
#   de CAETEC (<src>/<persona>/vacas/<ID robot>/foto.jpg), tomadas con
#   celular y a menudo con varias vacas en cuadro.
#   - Detector: YOLOv8 de COCO (clase "cow"); CowScanner está entrenado
#     en vista dorsal y no sirve para estas fotos.
#   - De las detecciones se queda con la de mayor (área × centralidad):
#     quien tomó la foto apuntó a la vaca del ID.
#   - Respeta la orientación EXIF de las fotos de celular.
#   - Ignora carpetas cuyo nombre no es un número (p.ej. "npi_VMS2").
#   - Salida: <dst>/<ID robot>/<persona>__<foto>.jpg y crops_log.csv.
#   - Con --person, <src> es plano (<src>/<ID robot>/foto.jpg, como
#     "IDs identificados como vigentes") y se usa ese nombre como persona.
#
# Uso (en la imagen muumetrics-tracking, que trae ultralytics):
#   python crop_caetec.py --src /data/Confirmacion_Caetec --dst /data/opencowid/caetec_crops \
#       --weights /weights/yolov8m.pt
# =============================================================

import argparse, csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

COW_CLASS = 19  # COCO


def pick_box(boxes, W, H):
    """Caja con mayor área relativa, penalizada por distancia al centro."""
    best, best_score = None, -1.0
    for (x1, y1, x2, y2), conf in boxes:
        area = (x2 - x1) * (y2 - y1) / (W * H)
        cx, cy = (x1 + x2) / 2 / W - 0.5, (y1 + y2) / 2 / H - 0.5
        score = area * (1 - min(1.0, (cx ** 2 + cy ** 2) ** 0.5))
        if score > best_score:
            best, best_score = (x1, y1, x2, y2, conf, area), score
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--weights", default="yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--person", default=None, help="Carpeta plana <src>/<ID>/foto; nombre a usar como persona")
    args = ap.parse_args()

    model = YOLO(args.weights)
    src, dst = Path(args.src), Path(args.dst)
    log = []
    for p in sorted(src.glob("*/*" if args.person else "*/vacas/*/*")):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        person, cow = (args.person, p.parts[-2]) if args.person else (p.parts[-4], p.parts[-2])
        row = {"persona": person, "id_robot": cow, "foto": p.name, "n_vacas": 0,
               "conf": "", "area_frac": "", "recorte": ""}
        if not cow.isdigit():
            row["recorte"] = "ignorada: ID no numérico"
            log.append(row)
            continue
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        W, H = im.size
        res = model(np.asarray(im)[:, :, ::-1], conf=args.conf, classes=[COW_CLASS], verbose=False)[0]
        boxes = [(b, float(c)) for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist())]
        row["n_vacas"] = len(boxes)
        if not boxes:
            row["recorte"] = "sin detección"
            log.append(row)
            continue
        x1, y1, x2, y2, conf, area = pick_box(boxes, W, H)
        mx, my = (x2 - x1) * args.margin, (y2 - y1) * args.margin
        box = (max(0, int(x1 - mx)), max(0, int(y1 - my)), min(W, int(x2 + mx)), min(H, int(y2 + my)))
        out = dst / cow / f"{person}__{p.stem}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        im.crop(box).save(out, quality=92)
        row.update(conf=round(conf, 3), area_frac=round(area, 3), recorte=str(out.relative_to(dst)))
        log.append(row)

    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "crops_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0]))
        w.writeheader(); w.writerows(log)
    ok = [r for r in log if r["conf"] != ""]
    print(f"Fotos: {len(log)} | recortadas: {len(ok)} | sin detección: "
          f"{sum(r['recorte'] == 'sin detección' for r in log)} | con >1 vaca: {sum(r['n_vacas'] > 1 for r in log)} | "
          f"IDs con recorte: {len({r['id_robot'] for r in ok})}")


if __name__ == "__main__":
    main()

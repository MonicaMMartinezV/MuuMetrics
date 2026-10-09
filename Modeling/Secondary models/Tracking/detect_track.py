#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: detect_track.py
# Descripción: Etapa 1 de re-identificación de vacas.
#   1) Detecta vacas en cada imagen con YOLO.
#   2) Agrupa las detecciones en "tracklets" (fotos consecutivas de la
#      misma cámara donde aparece la misma vaca). Un tracklet NO es una
#      vaca identificada: es un grupo de fotos que probablemente son del
#      mismo animal. La asignación de ID viene en la etapa 2.
#
# Salidas (en --out):
#   detections.csv : una fila por detección (imagen, caja, confianza, tracklet)
#   tracklets.csv  : un resumen por tracklet (cámara, inicio, fin, nº fotos)
#   crops/         : recortes de cada detección (para validación manual)
#
# Nombres de imagen esperados: AAAA-MM-DD-HH-MM-SS_camX_capY.jpg
# =============================================================

import argparse
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

NAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(cam\d+)_cap(\d+)\.jpg$", re.IGNORECASE)


def parse_name(name):
    """Devuelve (datetime, cámara) del nombre de archivo, o None si no coincide."""
    m = NAME_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d-%H-%M-%S"), m.group(2)


def run_detection(args, out_dir):
    model = YOLO(args.weights)
    print("Clases del modelo:", model.names)

    # Busca recursivamente: en Dataset/Images las fotos están dentro de
    # subcarpetas por BCS (2, 2.25, ...). La carpeta padre se guarda como bcs_folder.
    files = sorted(p for p in Path(args.images).rglob("*") if p.is_file() and parse_name(p.name))
    if args.limit:
        files = files[: args.limit]
    print(f"Imágenes a procesar: {len(files)}")

    crops_dir = out_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for start in range(0, len(files), args.batch):
        batch = files[start:start + args.batch]
        results = model.predict(
            [str(p) for p in batch],
            conf=args.conf,
            imgsz=args.imgsz,
            device=0,
            verbose=False,
        )
        for path, res in zip(batch, results):
            ts, cam = parse_name(path.name)
            img_h, img_w = res.orig_shape
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)

            image = None
            for (x1, y1, x2, y2), conf, cls in zip(boxes, confs, clss):
                if args.cow_class and res.names[int(cls)] != args.cow_class:
                    continue
                det_id = len(rows)
                row = {
                    "det_id": det_id,
                    "image": path.name,
                    "ts": ts,
                    "cam": cam,
                    "bcs_folder": path.parent.name,
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "conf": float(conf),
                    "cls": res.names[int(cls)],
                    "img_w": img_w, "img_h": img_h,
                }
                rows.append(row)

                if args.save_crops:
                    if image is None:
                        image = cv2.imread(str(path))
                    # recorte con 10% de margen alrededor de la caja
                    bw, bh = x2 - x1, y2 - y1
                    px, py = bw * 0.10, bh * 0.10
                    cx1 = int(max(0, x1 - px)); cy1 = int(max(0, y1 - py))
                    cx2 = int(min(img_w, x2 + px)); cy2 = int(min(img_h, y2 + py))
                    crop = image[cy1:cy2, cx1:cx2]
                    cv2.imwrite(str(crops_dir / f"{det_id:06d}_{path.stem}.jpg"), crop)

        print(f"  procesadas {min(start + args.batch, len(files))}/{len(files)}")

    return pd.DataFrame(rows)


def link_tracklets(df, max_gap_s, max_dist):
    """
    Enlaza detecciones entre fotos de la misma cámara.
    Una detección se une al tracklet abierto más cercano si:
      - la foto anterior de ese tracklet fue hace <= max_gap_s segundos,
      - la distancia entre centros, normalizada por el tamaño de la caja,
        es <= max_dist.
    Cada tracklet puede tener como máximo una detección por foto.
    """
    df = df.copy()
    df["tracklet_id"] = -1
    next_id = 0

    for cam, g_cam in df.groupby("cam"):
        open_tracks = []  # dicts con id, last_ts, last_image, last_cx, last_cy, last_diag
        for (ts, image), g_img in g_cam.sort_values(["ts", "image"]).groupby(["ts", "image"], sort=True):
            # descartar tracklets demasiado viejos
            open_tracks = [t for t in open_tracks if (ts - t["last_ts"]).total_seconds() <= max_gap_s]
            claimed = set()

            for idx, det in g_img.iterrows():
                cx = (det.x1 + det.x2) / 2
                cy = (det.y1 + det.y2) / 2
                diag = float(np.hypot(det.x2 - det.x1, det.y2 - det.y1))

                best, best_d = None, None
                for t in open_tracks:
                    if t["id"] in claimed:
                        continue
                    scale = max(diag, t["last_diag"], 1.0)
                    d = float(np.hypot(cx - t["last_cx"], cy - t["last_cy"])) / scale
                    if d <= max_dist and (best_d is None or d < best_d):
                        best, best_d = t, d

                if best is None:
                    best = {"id": next_id}
                    next_id += 1
                    open_tracks.append(best)

                claimed.add(best["id"])
                best.update(last_ts=ts, last_image=image, last_cx=cx, last_cy=cy, last_diag=diag)
                df.at[idx, "tracklet_id"] = best["id"]

    return df


def summarize(df):
    g = df.groupby("tracklet_id")
    summary = pd.DataFrame({
        "cam": g["cam"].first(),
        "inicio": g["ts"].min(),
        "fin": g["ts"].max(),
        "n_fotos": g["image"].nunique(),
        "n_detecciones": g.size(),
    }).reset_index()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Carpeta con las imágenes (AAAA-MM-DD-HH-MM-SS_camX_capY.jpg)")
    ap.add_argument("--weights", required=True, help="Pesos YOLO (.pt), p.ej. CowScanner.pt o yolov8n.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--cow_class", default=None,
                    help="Nombre de la clase de vaca a conservar (ej. 'cow'). Si no se pasa, se conservan todas las detecciones.")
    ap.add_argument("--max_gap_s", type=float, default=300.0,
                    help="Segundos máximos entre fotos para seguir un mismo tracklet")
    ap.add_argument("--max_dist", type=float, default=1.0,
                    help="Distancia máxima entre centros, en tamaños de caja")
    ap.add_argument("--save_crops", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Procesar solo las primeras N imágenes (prueba)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_detection(args, out_dir)
    if df.empty:
        print("No se detectó ninguna vaca.")
        return

    df = link_tracklets(df, args.max_gap_s, args.max_dist)
    summary = summarize(df)

    df.to_csv(out_dir / "detections.csv", index=False)
    summary.to_csv(out_dir / "tracklets.csv", index=False)

    print(f"Detecciones: {len(df)} | Fotos con vaca: {df['image'].nunique()} | Tracklets: {len(summary)}")
    print(f"Archivos en: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

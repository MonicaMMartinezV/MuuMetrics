#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: review_sheet.py
# Descripción: Arma la hoja para revisión manual de los IDs provisionales.
#   - Ordena los grupos por prioridad: primero los de menor cohesión
#     visual (más probable que estén mezclados), luego por tamaño.
#   - Genera montajes de recortes (40 por hoja) con el det_id visible.
#   - Genera dos CSV para llenar a mano:
#       review_cows.csv        : una fila por grupo (decisión general)
#       review_detections.csv  : una fila por recorte (para separar/mover)
#
# Valores sugeridos en review_detections.csv (columna revision):
#   ok        -> la vaca es el ID provisional
#   mover     -> es otra vaca; poner en nuevo_id un ID (existente o nuevo)
#   basura    -> no es vaca o la caja está mal
# =============================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from assign_ids import embed_crops

PER_SHEET = 40
COLS = 8
CELL = 120
LABEL_H = 14


def cohesion_table(df, feats):
    rows = []
    for cow_id, g in df.groupby("cow_id"):
        vecs = []
        for _, tg in g.groupby("tracklet_id"):
            v = np.mean([feats[d] for d in tg.det_id if d in feats], axis=0)
            vecs.append(v / np.linalg.norm(v))
        V = np.array(vecs)
        c = V.mean(axis=0)
        c /= np.linalg.norm(c)
        rows.append({
            "cow_id": int(cow_id),
            "n_detecciones": len(g),
            "n_fotos": g.image.nunique(),
            "n_tracklets": len(vecs),
            "cohesion": round(float((V @ c).mean()), 4),
        })
    t = pd.DataFrame(rows)
    # menor cohesión primero; a igualdad, los más grandes primero
    t = t.sort_values(["cohesion", "n_detecciones"], ascending=[True, False]).reset_index(drop=True)
    t.insert(0, "prioridad", np.arange(1, len(t) + 1))
    return t


def make_sheets(df, crops_dir, out_dir, prio):
    sheets_dir = out_dir / "hojas"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    files = {int(p.name.split("_")[0]): p for p in Path(crops_dir).glob("*.jpg")}
    hojas = {}
    for _, r in prio.iterrows():
        cow_id = int(r.cow_id)
        g = df[df.cow_id == cow_id].sort_values(["image", "det_id"])
        dets = g.det_id.tolist()
        parts = [dets[i:i + PER_SHEET] for i in range(0, len(dets), PER_SHEET)]
        names = []
        for p_idx, part in enumerate(parts, start=1):
            rows = int(np.ceil(len(part) / COLS))
            sheet = Image.new("RGB", (COLS * CELL, rows * (CELL + LABEL_H)), "white")
            draw = ImageDraw.Draw(sheet)
            for k, d in enumerate(part):
                if d not in files:
                    continue
                im = Image.open(files[d]).convert("RGB").resize((CELL, CELL))
                x, y = (k % COLS) * CELL, (k // COLS) * (CELL + LABEL_H)
                sheet.paste(im, (x, y + LABEL_H))
                draw.text((x + 2, y + 1), f"det {d}", fill="black")
            name = f"p{int(r.prioridad):02d}_id{cow_id:02d}_hoja{p_idx}.jpg"
            sheet.save(sheets_dir / name, quality=88)
            names.append(name)
        hojas[cow_id] = ", ".join(names)
    print(f"Hojas en: {sheets_dir}")
    return hojas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Carpeta con detections_ids.csv y crops/")
    args = ap.parse_args()

    run = Path(args.run)
    df = pd.read_csv(run / "detections_ids.csv")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feats = embed_crops(run / "crops", df.det_id.tolist(), device)
    prio = cohesion_table(df, feats)
    hojas = make_sheets(df, run / "crops", run / "review", prio)

    review_dir = run / "review"
    prio["hojas"] = prio.cow_id.map(hojas)
    prio["decision"] = ""       # aprobar / dividir / unir
    prio["unir_con_id"] = ""
    prio["notas"] = ""
    prio.to_csv(review_dir / "review_cows.csv", index=False)

    det = df[["det_id", "cow_id", "tracklet_id", "image", "cam", "ts"]].copy()
    det = det.merge(prio[["cow_id", "prioridad"]], on="cow_id").sort_values(["prioridad", "image"])
    det["revision"] = ""        # ok / mover / basura
    det["nuevo_id"] = ""        # solo si revision = mover
    det.to_csv(review_dir / "review_detections.csv", index=False)

    print(prio.head(8).to_string(index=False))
    print(f"\nGrupos: {len(prio)} | recortes: {len(det)} | hojas: {sum(len(v.split(', ')) for v in hojas.values())}")
    print(f"CSV para revisar: {review_dir / 'review_cows.csv'} y {review_dir / 'review_detections.csv'}")


if __name__ == "__main__":
    main()

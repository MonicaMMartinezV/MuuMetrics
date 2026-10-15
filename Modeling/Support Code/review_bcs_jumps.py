#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: review_bcs_jumps.py
# Descripción: Arma la hoja para revisión manual de los saltos de BCS
#   en dataset_del.csv: pares de fotos con el mismo vacaId, tomadas con
#   <= --max_days días de diferencia y con |ΔBCS| >= --min_jump. Un salto
#   así no es biológicamente posible, así que indica un ID mal asignado
#   (dos vacas distintas con el mismo vacaId) o un BCS mal etiquetado.
#   - Ordena los pares de mayor a menor salto.
#   - Genera montajes (PAIRS_PER_SHEET pares por hoja): foto A | foto B
#     con vacaId, fecha, BCS y flagMatch visibles.
#   - Genera review_saltos.csv para llenar a mano.
#
# Valores sugeridos en review_saltos.csv:
#   misma_vaca   : si / no / duda
#   bcs_correcto : a / b / ninguno / ambos  (cuál de las dos etiquetas es correcta)
#   bcs_sugerido : opcional, el BCS que pondrías a la foto mal etiquetada
#
# Uso:
#   python3 review_bcs_jumps.py --csv dataset_del.csv --images_dir Dataset/Images \
#       --out_dir Dataset/review_bcs_jumps
# =============================================================

import argparse, csv
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PAIRS_PER_SHEET = 4
CELL_W, CELL_H = 360, 400
HEADER_H = 30
LABEL_H = 54
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_jumps(rows, max_days, min_jump):
    by_cow = {}
    for r in rows:
        r["_ts"] = datetime.fromisoformat(r["foundDate"])
        by_cow.setdefault(r["vacaId"], []).append(r)
    pairs = []
    for cow_rows in by_cow.values():
        cow_rows.sort(key=lambda r: r["_ts"])
        # Pares consecutivos: cada foto contra la siguiente de la misma vaca
        for a, b in zip(cow_rows, cow_rows[1:]):
            days = (b["_ts"] - a["_ts"]).total_seconds() / 86400
            delta = float(b["BCS"]) - float(a["BCS"])
            if days <= max_days and abs(delta) >= min_jump:
                pairs.append((a, b, days, delta))
    pairs.sort(key=lambda p: (-abs(p[3]), p[0]["vacaId"], p[0]["_ts"]))
    return pairs


def load_font(size):
    # DejaVu trae acentos y "Δ"; la fuente por defecto de PIL no
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def fit(im, w, h):
    im.thumbnail((w, h))
    canvas = Image.new("RGB", (w, h), "white")
    canvas.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    return canvas


def make_sheets(pairs, index, out_dir):
    sheets_dir = out_dir / "hojas"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    font, font_b = load_font(15), load_font(19)
    row_h = HEADER_H + CELL_H + LABEL_H
    sheet_of = {}
    for s in range(0, len(pairs), PAIRS_PER_SHEET):
        part = pairs[s:s + PAIRS_PER_SHEET]
        sheet = Image.new("RGB", (2 * CELL_W + 10, len(part) * (row_h + 12)), "white")
        draw = ImageDraw.Draw(sheet)
        for k, (a, b, days, delta) in enumerate(part):
            n = s + k + 1
            y = k * (row_h + 12)
            draw.rectangle([0, y, sheet.width, y + HEADER_H], fill="#222")
            draw.text((8, y + 5), f"par {n:02d}   vaca {a['vacaId']}   ΔBCS {delta:+.2f} en {days:.1f} días",
                      fill="white", font=font_b)
            for j, r in enumerate((a, b)):
                x = j * (CELL_W + 10)
                path = index.get(r["ImgName"])
                cell = fit(Image.open(path).convert("RGB"), CELL_W, CELL_H) if path else \
                    Image.new("RGB", (CELL_W, CELL_H), "#ddd")
                sheet.paste(cell, (x, y + HEADER_H))
                ly = y + HEADER_H + CELL_H + 4
                flag = "  flagMatch" if r["flagMatch"] == "True" else ""
                draw.text((x + 4, ly), f"{'AB'[j]}  BCS {float(r['BCS']):.2f}   DEL {r['DEL']}{flag}",
                          fill="#b00" if flag else "black", font=font_b)
                draw.text((x + 4, ly + 25), f"{r['foundDate'][:16]}   {r['ImgName'][-9:]}",
                          fill="#555", font=font)
            sheet_of[n] = f"hoja{s // PAIRS_PER_SHEET + 1:02d}.jpg"
        sheet.save(sheets_dir / f"hoja{s // PAIRS_PER_SHEET + 1:02d}.jpg", quality=88)
    print(f"Hojas en: {sheets_dir}")
    return sheet_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset_del.csv")
    ap.add_argument("--images_dir", required=True, help="Carpeta con las imágenes (búsqueda recursiva por ImgName)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_days", type=float, default=3.0)
    ap.add_argument("--min_jump", type=float, default=1.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, newline="")))
    index = {p.stem: p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in IMG_EXTS}
    pairs = find_jumps(rows, args.max_days, args.min_jump)
    out_dir = Path(args.out_dir)
    sheet_of = make_sheets(pairs, index, out_dir)

    fields = ["par", "hoja", "vacaId", "dias", "delta_bcs",
              "img_a", "fecha_a", "bcs_a", "del_a", "flagMatch_a",
              "img_b", "fecha_b", "bcs_b", "del_b", "flagMatch_b",
              "misma_vaca", "bcs_correcto", "bcs_sugerido", "notas"]
    with open(out_dir / "review_saltos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for n, (a, b, days, delta) in enumerate(pairs, start=1):
            w.writerow({
                "par": n, "hoja": sheet_of[n], "vacaId": a["vacaId"],
                "dias": round(days, 2), "delta_bcs": delta,
                "img_a": a["ImgName"], "fecha_a": a["foundDate"], "bcs_a": a["BCS"],
                "del_a": a["DEL"], "flagMatch_a": a["flagMatch"],
                "img_b": b["ImgName"], "fecha_b": b["foundDate"], "bcs_b": b["BCS"],
                "del_b": b["DEL"], "flagMatch_b": b["flagMatch"],
                "misma_vaca": "", "bcs_correcto": "", "bcs_sugerido": "", "notas": "",
            })
    cows = {a["vacaId"] for a, *_ in pairs}
    flagged = sum(a["flagMatch"] == "True" or b["flagMatch"] == "True" for a, b, *_ in pairs)
    print(f"Pares: {len(pairs)} | vacas: {len(cows)} | con flagMatch: {flagged} | hojas: {len(set(sheet_of.values()))}")
    print(f"CSV para revisar: {out_dir / 'review_saltos.csv'}")


if __name__ == "__main__":
    main()

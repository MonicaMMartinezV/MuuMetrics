#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: review_byid_jumps.py
# Descripción: Hojas de revisión por vaca para los grupos de ByID
#   (<ids_dir>/<vaca>/img.jpg) con saltos de BCS imposibles: dos fotos
#   de la misma vaca con |ΔBCS| >= --min_jump en <= --max_days días.
#   - El BCS sale de <labels_dir>/<BCS>/img.jpg (mismo nombre de archivo).
#   - Una hoja por vaca con TODAS sus fotos ordenadas por fecha; las
#     fotos que participan en algún salto van con borde rojo.
#   - revision_fotos.csv: una fila por foto para llenar a mano.
#
# Valores sugeridos en revision_fotos.csv:
#   es_esta_vaca : si / no / duda   (¿la foto pertenece al grupo?)
#   bcs_sugerido : BCS correcto si la etiqueta está mal (vacío = está bien)
#
# Uso:
#   python3 review_byid_jumps.py --ids_dir Dataset/ByID --labels_dir Dataset/Images \
#       --out_dir Dataset/review_byid_jumps
# =============================================================

import argparse, csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from review_bcs_jumps import fit, load_font

COLS = 5
CELL_W, CELL_H = 300, 330
LABEL_H = 48
HEADER_H = 44
BORDER = 6


def load_cows(ids_dir: Path, labels_dir: Path):
    labels = {}
    for p in labels_dir.rglob("*.jpg"):
        try:
            labels.setdefault(p.name, float(p.parent.name))
        except ValueError:
            pass
    cows = defaultdict(dict)
    for p in ids_dir.glob("*/*.jpg"):
        if not p.parent.name.isdigit():
            continue
        name = p.name.replace(" - Copy", "")
        if name in labels and name not in cows[p.parent.name]:
            ts = datetime.strptime(name[:19], "%Y-%m-%d-%H-%M-%S")
            cows[p.parent.name][name] = {"path": p, "ts": ts, "bcs": labels[name]}
    return {c: sorted(d.items(), key=lambda kv: kv[1]["ts"]) for c, d in cows.items()}


def find_jumps(photos, max_days, min_jump):
    """Pares (a, b, días, Δ) con salto; a antes que b."""
    pairs = []
    for i, (na, a) in enumerate(photos):
        for nb, b in photos[i + 1:]:
            days = (b["ts"] - a["ts"]).total_seconds() / 86400
            if days <= max_days and abs(b["bcs"] - a["bcs"]) >= min_jump:
                pairs.append((na, nb, days, b["bcs"] - a["bcs"]))
    return pairs


def make_sheet(cow, photos, pairs, out_path):
    in_jump = {n for a, b, *_ in pairs for n in (a, b)}
    rows = (len(photos) + COLS - 1) // COLS
    cell_h = CELL_H + LABEL_H
    sheet = Image.new("RGB", (COLS * (CELL_W + 8) + 8, HEADER_H + rows * (cell_h + 8) + 8), "white")
    draw = ImageDraw.Draw(sheet)
    font, font_b, font_h = load_font(14), load_font(17), load_font(21)
    worst = max(pairs, key=lambda p: abs(p[3]))
    draw.rectangle([0, 0, sheet.width, HEADER_H], fill="#222")
    draw.text((10, 10), f"vaca {cow}   {len(photos)} fotos   {len(pairs)} pares con salto   "
                        f"máx ΔBCS {worst[3]:+.2f} en {worst[2]:.2f} días   (rojo = foto en algún salto)",
              fill="white", font=font_h)
    for k, (name, r) in enumerate(photos):
        x = 8 + (k % COLS) * (CELL_W + 8)
        y = HEADER_H + 8 + (k // COLS) * (cell_h + 8)
        if name in in_jump:
            draw.rectangle([x - 4, y - 4, x + CELL_W + 3, y + cell_h + 3], fill="#d00")
        sheet.paste(fit(Image.open(r["path"]).convert("RGB"), CELL_W, CELL_H), (x, y))
        draw.rectangle([x, y + CELL_H, x + CELL_W - 1, y + cell_h - 1], fill="white")
        draw.text((x + 4, y + CELL_H + 3), f"#{k + 1:02d}   BCS {r['bcs']:.2f}", fill="black", font=font_b)
        draw.text((x + 4, y + CELL_H + 25), f"{r['ts']:%d %b %H:%M:%S}   {name[20:-4]}", fill="#555", font=font)
    sheet.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_days", type=float, default=3.0)
    ap.add_argument("--min_jump", type=float, default=1.0)
    args = ap.parse_args()

    cows = load_cows(Path(args.ids_dir), Path(args.labels_dir))
    out_dir = Path(args.out_dir)
    (out_dir / "hojas").mkdir(parents=True, exist_ok=True)
    fields = ["vaca", "hoja", "n", "img", "fecha", "bcs", "en_salto", "es_esta_vaca", "bcs_sugerido", "notas"]
    flagged = []
    with open(out_dir / "revision_fotos.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cow in sorted(cows, key=int):
            pairs = find_jumps(cows[cow], args.max_days, args.min_jump)
            if not pairs:
                continue
            flagged.append((cow, len(cows[cow]), len(pairs)))
            sheet_name = f"vaca_{int(cow):02d}.jpg"
            make_sheet(cow, cows[cow], pairs, out_dir / "hojas" / sheet_name)
            in_jump = {n for a, b, *_ in pairs for n in (a, b)}
            for k, (name, r) in enumerate(cows[cow], start=1):
                w.writerow({"vaca": cow, "hoja": sheet_name, "n": k, "img": name,
                            "fecha": r["ts"].isoformat(sep=" "), "bcs": r["bcs"],
                            "en_salto": "si" if name in in_jump else "",
                            "es_esta_vaca": "", "bcs_sugerido": "", "notas": ""})
    for cow, n, p in flagged:
        print(f"vaca {cow:>3}: {n:2d} fotos, {p:2d} pares con salto")
    print(f"{len(flagged)} hojas en {out_dir / 'hojas'} | CSV: {out_dir / 'revision_fotos.csv'}")


if __name__ == "__main__":
    main()

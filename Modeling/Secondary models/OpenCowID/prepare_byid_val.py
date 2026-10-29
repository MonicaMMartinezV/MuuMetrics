#!/usr/bin/env python3
# =============================================================
# Nombre del archivo: prepare_byid_val.py
# Descripción: Arma el par galería/consulta de OpenCowID con las vacas
#   de ByID (<ids_dir>/<vaca>/img.jpg), para validar el entrenamiento
#   con fotos reales de vista dorsal.
#   - Solo vacas con fotos en >= 2 días distintos.
#   - Galería = fotos del primer día; consulta = fotos de los días
#     siguientes. Así se mide reconocer a la vaca en otro día, no en
#     la foto de al lado del mismo evento.
#   - Galería y consulta tienen las mismas carpetas (OpenCowID usa
#     ImageFolder: el índice de clase sale del orden de las carpetas).
#
# Uso:
#   python3 prepare_byid_val.py --ids_dir Dataset/ByID --out Dataset/opencowid/byid_val
# =============================================================

import argparse, shutil
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    n_cows = n_gal = n_query = 0
    for cow in sorted(Path(args.ids_dir).iterdir()):
        if not (cow.is_dir() and cow.name.isdigit()):
            continue
        by_day = defaultdict(list)
        for p in cow.glob("*.jpg"):
            if " - Copy" not in p.name:
                by_day[p.name[:10]].append(p)
        if len(by_day) < 2:
            continue
        days = sorted(by_day)
        for split, ps in (("gallery", by_day[days[0]]), ("query", [p for d in days[1:] for p in by_day[d]])):
            (out / split / cow.name).mkdir(parents=True, exist_ok=True)
            for p in ps:
                shutil.copy2(p, out / split / cow.name / p.name)
        n_cows += 1
        n_gal += len(by_day[days[0]])
        n_query += sum(len(by_day[d]) for d in days[1:])
    print(f"{n_cows} vacas | galería: {n_gal} fotos | consulta: {n_query} fotos -> {out}")


if __name__ == "__main__":
    main()

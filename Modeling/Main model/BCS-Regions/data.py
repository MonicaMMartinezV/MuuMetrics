# =============================================================
# Nombre del archivo: data.py
#
# Descripción:
#   Lectura de Dataset_Fr_BCS: carpetas <vacaId>_<BCS>/ con los
#   recortes dorsales del robot (ByIDxx__AAAA-MM-DD-hh-mm-ss_camX_capY.jpg).
#   La vaca y el BCS salen del nombre de la carpeta (confirmados a mano).
#   - "<id>_MAYO4.25_JUNIO4_JULIO3.5": BCS por mes, según la fecha del
#     nombre de archivo.
#   - Carpetas que no se pueden leer (p.ej. dos vacas mezcladas,
#     "1552_Vaca1es2.5_Vaca2es4.75") se ignoran y se reportan.
# =============================================================
from pathlib import Path
from typing import Dict, List

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MONTHS = {"ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6, "JULIO": 7,
          "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12}

def load_records(data_dir: str, verbose: bool = True) -> List[Dict]:
    """-> [{"path", "name", "vaca", "bcs"}] ordenados por carpeta y nombre."""
    records, skipped = [], []
    for cow_dir in sorted(d for d in Path(data_dir).iterdir() if d.is_dir()):
        cow, _, spec = cow_dir.name.partition("_")
        imgs = sorted(p for p in cow_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
        by_month = {}
        for tok in spec.split("_"):
            name = tok.rstrip("0123456789.")
            if name.upper() in MONTHS:
                by_month[MONTHS[name.upper()]] = float(tok[len(name):])
        try:
            fixed = None if by_month else float(spec)
        except ValueError:
            skipped.append(f"{cow_dir.name} ({len(imgs)} imgs)")
            continue
        for p in imgs:
            bcs = fixed if fixed is not None else by_month.get(int(p.stem.split("__")[-1][5:7]))
            if bcs is None:
                skipped.append(f"{cow_dir.name}/{p.name} (mes sin BCS)")
                continue
            records.append({"path": str(p), "name": f"{cow_dir.name}/{p.name}", "vaca": cow, "bcs": bcs})
    if verbose:
        print(f"[datos] {len(records)} imágenes, {len({r['vaca'] for r in records})} vacas | ignoradas: {skipped}")
    if not records:
        raise RuntimeError(f"No se encontró ninguna imagen en {data_dir}.")
    return records

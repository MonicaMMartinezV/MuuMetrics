# =============================================================
# Nombre del archivo: del_data.py
#
# Descripción:
#   DEL (días en leche) de cada foto, leído de <work>/del.csv
#   (generado con del/build_del.py a partir de GeneralidadesOrdeno).
#   Se cruza por el nombre de la foto ('carpeta/ByIDxx__foto.jpg', igual
#   que en features.npz). El BCS NO se toma de aquí: sale del nombre de
#   la carpeta (data.py). Foto sin DEL -> NaN.
# =============================================================
import csv
from pathlib import Path
import numpy as np

def load_del(names, work_dir="/work"):
    """-> (DEL [n], lactación [n]) alineados con names, NaN si falta."""
    f = Path(work_dir) / "del.csv"
    if not f.exists(): raise SystemExit(f"No existe {f}: corre primero del/build_del.py")
    rows = {r["foto"]: r for r in csv.DictReader(open(f))}
    d, lac = np.full(len(names), np.nan), np.full(len(names), np.nan)
    for i, n in enumerate(names):
        r = rows.get(str(n))
        if r and r["DEL"]: d[i] = float(r["DEL"])
        if r and r["lactacion"]: lac[i] = float(r["lactacion"])
    return d, lac

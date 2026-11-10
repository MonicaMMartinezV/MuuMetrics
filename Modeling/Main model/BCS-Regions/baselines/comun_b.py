# =============================================================
# Nombre del archivo: baselines/comun_b.py
#
# Descripción:
#   Piezas compartidas por los baselines, para que todos compitan con el
#   modelo (bcs_regions.py) en las mismas condiciones:
#     - mismos datos (Dataset_Fr_BCS_v2, vaca y BCS desde la carpeta);
#     - mismas particiones: división "vaca" = las semillas 0..n-1 del
#       esquema 5 folds por vaca de 'evaluate' (ablaciones/comun.splits_for);
#       división "foto" = 5 folds estratificados por foto, ignorando la vaca
#       (el protocolo con fuga de identidad de buena parte de la literatura);
#     - validación interna para elegir hiperparámetros DENTRO del train
#       (20 % del train, por vaca o por foto según la división);
#     - mismo formato de predicciones fuera de fold (preds_seed<s>.csv).
# =============================================================
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ)); sys.path.insert(0, str(RAIZ / "ablaciones"))
import bcs_regions as br                      # noqa: E402
from data import load_records                 # noqa: E402
from comun import splits_for, fit_apply       # noqa: E402

DIVISIONES = ("vaca", "foto")

def cargar(data_dir):
    recs = load_records(data_dir)
    return (recs, np.array([r["bcs"] for r in recs]), np.array([r["vaca"] for r in recs]),
            np.array([r["name"] for r in recs]))

def estratos(y): return np.round(y * 2).astype(int)

def particiones(y, cows, division, semillas):
    """Lista (una por semilla) de listas de (train, test)."""
    if division == "vaca": return splits_for(y, cows, semillas)["rep5x10"]
    return [list(StratifiedKFold(5, shuffle=True, random_state=s).split(y, estratos(y))) for s in range(semillas)]

def interna(tr, y, cows, division, seed, intentos=30):
    """Parte el train en ajuste (~80 %) y validación (~20 %) para elegir lr y época.
    Con ~50 vacas, un 20 % por vaca puede salir muy sesgado (p. ej. solo vacas gordas), y entonces
    elegir con él premia predecir cerca de su media. Por eso, entre `intentos` particiones al azar
    se queda con la que mejor reproduce la media y la desviación del BCS del train. Solo usa el train."""
    mejor = None
    for k in range(intentos):
        rs = seed * 1000 + k
        if division == "vaca":
            a, v = next(StratifiedGroupKFold(5, shuffle=True, random_state=rs).split(tr, estratos(y[tr]), cows[tr]))
        else:
            a, v = next(StratifiedKFold(5, shuffle=True, random_state=rs).split(tr, estratos(y[tr])))
        ya, yv = y[tr[a]], y[tr[v]]; d = abs(ya.mean() - yv.mean()) + abs(ya.std() - yv.std())
        if mejor is None or d < mejor[0]: mejor = (d, tr[a], tr[v])
    return mejor[1], mejor[2]

def guardar_preds(path: Path, names, cows, y, p):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["foto", "vaca", "bcs_real", "bcs_pred"])
        w.writerows([[n, c, yy, round(float(pp), 4)] for n, c, yy, pp in zip(names, cows, y, p)])

def leer_preds(path: Path):
    with open(path) as fh: rows = list(csv.DictReader(fh))
    return np.array([r["foto"] for r in rows]), np.array([float(r["bcs_pred"]) for r in rows])

# =============================================================
# Nombre del archivo: ablaciones/comun.py
#
# Descripción:
#   Piezas compartidas por las ablaciones. Todo sale de bcs_regions.py
#   (model/main), así las ablaciones siguen al modelo si cambia el
#   backbone, el bloque o las regiones, sin tocar los scripts:
#     splits_for   esquemas de validación de 'evaluate' (LOCO y 5 folds x semillas)
#     fit_apply    ridge del modelo (br.fit_predict) -> predicciones
#     tokens       CLS y tokens de parche del modelo (br.patch_tokens)
#     grid         malla de parches del backbone
#     modelo       carga features.npz comprobando su firma y devuelve la
#                  estructura del descriptor (regiones y dimensión D)
#     cache_ok / guardar_cache
#                  las cachés de descriptores de cada ablación guardan la
#                  firma del modelo; si no coincide con la de features.npz
#                  se recalculan solas
# =============================================================
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br

def splits_for(y, cows, seeds):
    strata = np.round(y * 2).astype(int)
    return {"loco": [list(GroupKFold(len(np.unique(cows))).split(y, y, cows))],
            "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(seeds)]}

def fit_apply(Xtr, ytr, gtr, Xte):
    return br.fit_predict(Xtr, ytr, gtr, Xte)[0]

def tokens(bb, img, dev):
    """img RGB -> (CLS [D], parches [g*g, D]) exactamente como br.describe()."""
    cls, patches, _ = br.patch_tokens(bb, img, dev)
    return cls, patches

def grid(bb) -> int:
    return br.SIZE // bb.patch_embed.patch_size[0]

def modelo(work: Path):
    """features.npz del modelo actual -> (X, y, vacas, nombres, firma, regiones, D). Se detiene si es de otro modelo."""
    br.check_features(work)
    X, y, cows, names = br.load_features(work); f = br.load_firma(work) or br.firma(work)
    R = len(f["regions"]); D = (X.shape[1] - R) // R
    assert R * D + R == X.shape[1], f"features.npz ({X.shape[1]} columnas) no cuadra con {R} regiones"
    return X, y, cows, names, f, list(f["regions"]), D

def cache_ok(path: Path, f: dict) -> bool:
    if not path.exists(): return False
    z = np.load(path)
    ok = "firma" in z.files and json.loads(str(z["firma"])) == f
    if not ok: print(f"[caché] {path.name} es de otro modelo (o sin firma): se recalcula", flush=True)
    return ok

def guardar_cache(path: Path, f: dict, **arrays):
    np.savez_compressed(path, firma=json.dumps(f), **arrays)

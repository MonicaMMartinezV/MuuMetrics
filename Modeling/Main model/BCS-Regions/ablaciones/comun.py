# =============================================================
# Nombre del archivo: ablaciones/comun.py
#
# Descripción:
#   Piezas compartidas por las ablaciones, tomadas del modelo actual
#   (bcs_regions.py de model/main: DINOv3-L/16 congelado, bloque br.LAYER
#   + LayerNorm final, ridge con alpha elegido por GroupKFold(4) por vaca,
#   sin calibración). Así cada ablación cambia solo lo que estudia.
#     splits_for  esquemas de validación de 'evaluate' (LOCO y 5 folds x semillas)
#     fit_apply   ridge del modelo (br.fit_predict) -> predicciones
#     tokens      CLS y tokens de parche del bloque del modelo + tamaño de malla
# =============================================================
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

import bcs_regions as br
from segmenter import MEAN, STD

def splits_for(y, cows, seeds):
    strata = np.round(y * 2).astype(int)
    return {"loco": [list(GroupKFold(len(np.unique(cows))).split(y, y, cows))],
            "rep5x10": [list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(y, strata, cows)) for s in range(seeds)]}

def fit_apply(Xtr, ytr, gtr, Xte):
    return br.fit_predict(Xtr, ytr, gtr, Xte)[0]

def grid(bb) -> int:
    return br.SIZE // bb.patch_embed.patch_size[0]

@torch.no_grad()
def tokens(bb, img, dev):
    """img RGB -> (CLS [1024], parches [g*g, 1024]) del bloque br.LAYER con la LayerNorm final, igual que br.describe()."""
    x = np.asarray(Image.fromarray(np.ascontiguousarray(img)).resize((br.SIZE, br.SIZE), Image.BICUBIC)) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1)).float()[None].to(dev)
    with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
        sp, pre = bb.forward_intermediates(x, indices=[br.LAYER], norm=True, return_prefix_tokens=True,
                                           output_fmt="NCHW", intermediates_only=True)[0]
    return pre[0, 0].float(), sp[0].float().reshape(sp.shape[1], -1).T

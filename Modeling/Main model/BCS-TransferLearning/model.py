# =============================================================
# Nombre del archivo: model.py
#
# Descripción:
#   BEiT-B/16 (misma arquitectura que el modelo de Winkler et al. 2024:
#   BeitForImageClassification sobre microsoft/beit-base-patch16-224)
#   adaptado a regresión de BCS con las 6 máscaras de región:
#     - Bloques 0..11-K congelados (sin gradiente); solo los últimos K
#       bloques se ajustan.
#     - Las máscaras entran como un embedding de región (Linear 6->768,
#       inicializado en cero) que se suma a los tokens justo antes del
#       primer bloque entrenable: al inicio el modelo es idéntico al
#       preentrenado.
#     - Cabeza: promedio de los tokens finales dentro de cada región
#       (6x768) + promedio global (768) -> dropout -> Linear -> BCS.
#   SimMIM: preentrenamiento autosupervisado (reconstruir los píxeles de
#   los parches ocultos) para entrenar el backbone desde cero o adaptarlo
#   a las fotos del robot sin usar etiquetas de BCS.
# =============================================================
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import BeitConfig, BeitModel

HF_ID = "microsoft/beit-base-patch16-224"     # punto de partida de Winkler (ImageNet-22k -> 1k)
SIZE, PATCH = 224, 16
GRID = SIZE // PATCH                          # 14x14 parches
N_REG = 6

def load_beit(spec: str, drop_path: float = 0.1, mask_token: bool = False) -> BeitModel:
    """spec:
      'imagenet'          pesos de microsoft/beit-base-patch16-224
      'scratch'           misma arquitectura, pesos aleatorios
      <carpeta>           salida de 'pretrain' (save_pretrained)
      <archivo .ckpt/.pt> state_dict de BeitForImageClassification (p.ej. el checkpoint
                          Lightning de Winkler, con prefijo 'vit.beit.'); el classifier se ignora
    """
    kw = dict(drop_path_rate=drop_path, use_mask_token=mask_token)
    if spec == "imagenet":
        return BeitModel.from_pretrained(HF_ID, **kw)
    if spec == "scratch":
        return BeitModel(BeitConfig.from_pretrained(HF_ID, **kw))
    p = Path(spec)
    if p.is_dir():
        return BeitModel.from_pretrained(p, **kw)
    if p.is_file():
        sd = torch.load(p, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        out = {}
        for k, v in sd.items():
            for pre in ("vit.beit.", "model.beit.", "beit."):
                if k.startswith(pre): out[k[len(pre):]] = v; break
        net = BeitModel(BeitConfig.from_pretrained(HF_ID, **kw))
        missing, unexpected = net.load_state_dict(out, strict=False)
        missing = [k for k in missing if k != "embeddings.mask_token"]
        print(f"[backbone] {p.name}: {len(out)} tensores, faltan {len(missing)}, sobran {len(unexpected)}")
        if missing: raise SystemExit(f"El checkpoint no corresponde a BEiT-B (faltan p.ej. {missing[:3]})")
        return net
    raise SystemExit(f"Backbone desconocido: {spec}")

def run_blocks(beit: BeitModel, h: torch.Tensor, blocks) -> torch.Tensor:
    enc = beit.encoder
    for blk in blocks:
        rpb = (enc.relative_position_bias((GRID, GRID), interpolate_pos_encoding=False, dim_size=h.shape[1])
               if enc.has_relative_position_bias else None)
        h = blk(h, relative_position_bias=rpb, resolution=(SIZE, SIZE))[0]
    return h

class BCSBeit(nn.Module):
    def __init__(self, beit: BeitModel, n_train: int, y_mean: float, head_drop: float = 0.3):
        super().__init__()
        self.beit, d = beit, beit.config.hidden_size
        self.n_frozen = beit.config.num_hidden_layers - n_train
        self.region = nn.Linear(N_REG, d); nn.init.zeros_(self.region.weight); nn.init.zeros_(self.region.bias)
        pool_ln = beit.pooler.layernorm if beit.pooler is not None and beit.pooler.layernorm is not None else None
        self.norm = nn.LayerNorm(d) if pool_ln is None else pool_ln
        self.head = nn.Sequential(nn.Dropout(head_drop), nn.Linear((N_REG + 1) * d, 1))
        nn.init.zeros_(self.head[1].weight); nn.init.constant_(self.head[1].bias, y_mean)
        for p in beit.parameters(): p.requires_grad = False
        for blk in self.trainable_blocks():
            for p in blk.parameters(): p.requires_grad = True

    def trainable_blocks(self):
        return list(self.beit.encoder.layer[self.n_frozen:])

    def train(self, mode: bool = True):
        super().train(mode)
        self.beit.embeddings.eval()                       # el prefijo congelado siempre en eval (sin drop-path)
        for blk in self.beit.encoder.layer[:self.n_frozen]: blk.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, 3+6, 224, 224]: RGB normalizado + máscaras en [0,1] -> BCS [B]."""
        rgb, m = x[:, :3], x[:, 3:]
        frac = F.adaptive_avg_pool2d(m, GRID).flatten(2).transpose(1, 2)            # [B,196,6] fracción de cada parche
        with torch.no_grad():
            h, _ = self.beit.embeddings(rgb)
            h = run_blocks(self.beit, h, self.beit.encoder.layer[:self.n_frozen])
        h = torch.cat([h[:, :1], h[:, 1:] + self.region(frac)], 1)
        h = run_blocks(self.beit, h, self.trainable_blocks())
        tok = self.norm(h[:, 1:])                                                    # [B,196,768]
        tot = frac.sum(1)                                                            # [B,6] parches por región
        reg = torch.einsum("bpr,bpd->brd", frac, tok) / tot.clamp_min(1e-3)[..., None]
        reg = reg * (tot >= 0.5)[..., None]                                          # región ausente -> 0
        feat = torch.cat([reg.flatten(1), tok.mean(1)], 1)
        return self.head(feat).squeeze(1)

    def param_groups(self, lr_head: float, lr_bb: float, llrd: float, wd: float):
        groups = [{"params": list(self.region.parameters()) + list(self.head.parameters()), "lr": lr_head, "weight_decay": wd, "base_lr": lr_head}]
        norm_p = [p for p in self.norm.parameters()]
        blocks = self.trainable_blocks()
        for i, blk in enumerate(blocks):                                             # LLRD: el último bloque con lr_bb
            lr = lr_bb * llrd ** (len(blocks) - 1 - i)
            groups.append({"params": list(blk.parameters()), "lr": lr, "weight_decay": wd, "base_lr": lr})
        if blocks:
            for p in norm_p: p.requires_grad = True
            groups.append({"params": norm_p, "lr": lr_bb, "weight_decay": 0.0, "base_lr": lr_bb})
        return groups

class SimMIM(nn.Module):
    """Reconstrucción de píxeles (normalizados por parche) de los parches ocultos, pérdida L1."""
    def __init__(self, beit: BeitModel):
        super().__init__()
        self.beit = beit
        self.dec = nn.Linear(beit.config.hidden_size, PATCH * PATCH * 3)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h, _ = self.beit.embeddings(x, bool_masked_pos=mask)
        h = run_blocks(self.beit, h, self.beit.encoder.layer)
        pred = self.dec(h[:, 1:])                                                    # [B,196,768]
        tgt = F.unfold(x, PATCH, stride=PATCH).transpose(1, 2)                       # [B,196,3*16*16]
        tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()
        return ((pred - tgt).abs().mean(-1) * mask).sum() / mask.sum().clamp_min(1)

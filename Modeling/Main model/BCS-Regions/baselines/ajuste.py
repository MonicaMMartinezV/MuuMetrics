# =============================================================
# Nombre del archivo: baselines/ajuste.py
#
# Descripción:
#   Baselines AJUSTADOS de punta a punta (lo que hace la literatura 2D),
#   entrenados y evaluados con el mismo protocolo que el modelo:
#     resnet50         ResNet-50 (timm resnet50.a1_in1k), regresión
#     efficientnet_b0  EfficientNet-B0 (timm efficientnet_b0.ra_in1k), regresión
#     efficientnetv2_s EfficientNetV2-S (timm tf_efficientnetv2_s.in1k), regresión
#     vit_b16          ViT-B/16 supervisado (DeiT-B 384, ImageNet-1k), regresión
#     dinov3_lora      el mismo DINOv3-L/16 del modelo, ajustado con LoRA
#                      (rango 8 en qkv y proyección de los 24 bloques) +
#                      cabeza lineal; responde "¿y si se ajusta el modelo
#                      fundacional en vez de congelarlo?"
#   Entrada: foto aplastada a 384 x 384 (ENTRADA, fija aunque el modelo use otra), normalización
#   ImageNet. Aumentos: volteo horizontal, afín leve (±10°, ±5 %, escala
#   0.9-1.1) y color (brillo/contraste 0.2, saturación 0.1).
#   Objetivo: BCS estandarizado con media/desv. del train, pérdida
#   SmoothL1, AdamW (wd 0.05), calentamiento 1 época + coseno, fp16.
#   Selección justa de hiperparámetros DENTRO del train de cada fold:
#   para cada lr de la rejilla se entrena en el 80 % del train y se elige
#   la época con menor MAE en el 20 % restante (por vaca o por foto, según
#   la división); el lr con menor MAE de validación predice el test.
#
#   Salidas: <work>/baselines/<modelo>_<division>/
#            fold_s<semilla>_k<fold>.npz (caché, permite reanudar; guarda la
#              configuración y se recalcula si cambia, p. ej. el backbone de main),
#            preds_seed<s>.csv, log.csv (lr, época y MAE de validación por fold)
#
# Uso: docker compose -f baselines/docker-compose.yml run --rm baselines \
#        baselines/ajuste.py --modelo resnet50 --division vaca
#      --rapido: 1 fold, 2 épocas, 1 lr (prueba de humo)
# =============================================================
import argparse, csv, json, math, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image
from torchvision.transforms import v2

from comun_b import br, cargar, particiones, interna, guardar_preds, DIVISIONES, ENTRADA

MODELOS = {
    "resnet50":        dict(timm="resnet50.a1_in1k",        lrs=(1e-4, 3e-4, 1e-3), epocas=30, batch=16),
    "efficientnet_b0": dict(timm="efficientnet_b0.ra_in1k", lrs=(3e-4, 1e-3),       epocas=30, batch=16),   # 2 lr por costo (~4.5 h -> ~3 h)
    "dinov3_lora":     dict(timm=br.BACKBONE,               lrs=(3e-4, 1e-3),       epocas=15, batch=8, lora=8),
    "efficientnetv2_s": dict(timm="tf_efficientnetv2_s.in1k",  lrs=(3e-4, 1e-3),       epocas=30, batch=16),   # Siachos et al. 2023 usan EfficientNetV2
    "vit_b16":         dict(timm="deit_base_patch16_384.fb_in1k", lrs=(3e-5, 1e-4),    epocas=15, batch=8, ckpt=True),   # ViT supervisado (ImageNet-1k, 384 nativo)
}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AUG = v2.Compose([v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                  v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)])

class LoRA(nn.Module):
    """y = W x + (B A x) * alpha/r, con W congelada; B empieza en 0 (al inicio es el modelo original)."""
    def __init__(self, base: nn.Linear, r: int, alpha: float = None):
        super().__init__(); self.base, self.scale = base, (alpha or r) / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(self, x): return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

def crear(cfg):
    if "lora" in cfg:
        m = timm.create_model(cfg["timm"], pretrained=True, num_classes=1, img_size=ENTRADA)
        for p in m.parameters(): p.requires_grad = False
        for b in m.blocks: b.attn.qkv, b.attn.proj = LoRA(b.attn.qkv, cfg["lora"]), LoRA(b.attn.proj, cfg["lora"])
        for p in m.head.parameters(): p.requires_grad = True
        m.set_grad_checkpointing(True)
        return m
    m = timm.create_model(cfg["timm"], pretrained=True, num_classes=1)
    if cfg.get("ckpt"): m.set_grad_checkpointing(True)
    return m

def cargar_imagenes(recs):
    """Todas las fotos aplastadas a SIZE x SIZE (como el modelo), uint8 [N,3,S,S]."""
    return torch.from_numpy(np.stack([np.asarray(Image.open(r["path"]).convert("RGB").resize((ENTRADA, ENTRADA), Image.BICUBIC))
                                      for r in recs]).transpose(0, 3, 1, 2).copy())

def lote(X, idx, dev, aug):
    x = torch.stack([AUG(X[i]) for i in idx]) if aug else X[idx]
    return ((x.float() / 255.0 - MEAN) / STD).to(dev, non_blocking=True)

@torch.no_grad()
def predecir(m, X, idx, dev, bs=32):
    m.eval(); out = []
    for i in range(0, len(idx), bs):
        with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
            out.append(m(lote(X, idx[i:i + bs], dev, False)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()

def entrenar(cfg, X, y, ia, iv, lr, epocas, seed, dev):
    """Entrena en ia, valida cada época en iv; devuelve el modelo de la mejor época, su MAE y la época (1-based)."""
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    m = crear(cfg).to(dev); params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05); bs = cfg["batch"]
    pasos_ep = math.ceil(len(ia) / bs); total = epocas * pasos_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / pasos_ep) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    mu, sd = float(y[ia].mean()), float(y[ia].std()); yt = torch.tensor((y - mu) / sd, dtype=torch.float32)
    entrenables = {n for n, p in m.named_parameters() if p.requires_grad}
    mejor = (math.inf, None, 0)
    for ep in range(epocas):
        m.train(); perm = rng.permutation(ia)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2: continue
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                loss = F.smooth_l1_loss(m(lote(X, b, dev, True)).float().squeeze(1), yt[b].to(dev))
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        mae = float(np.abs(predecir(m, X, iv, dev) * sd + mu - y[iv]).mean())
        if mae < mejor[0]:
            st = m.state_dict(); bufs = {n for n, _ in m.named_buffers()}
            mejor = (mae, {k: v.detach().cpu().clone() for k, v in st.items() if k in entrenables or k in bufs}, ep + 1)
    m.load_state_dict(mejor[1], strict=False)
    return m, mejor[0], mejor[2], (mu, sd)

def main(a):
    cfg = MODELOS[a.modelo]; dev = br.device(); recs, y, cows, names = cargar(a.data_dir)
    out = Path(a.work_dir) / "baselines" / f"{a.modelo}_{a.division}{'_rapido' if a.rapido else ''}"; out.mkdir(parents=True, exist_ok=True)
    lrs, epocas = (cfg["lrs"][:1], 2) if a.rapido else (cfg["lrs"], cfg["epocas"])
    print(f"[{a.modelo} / división {a.division}] {len(y)} fotos, {len(np.unique(cows))} vacas | lrs {lrs} | {epocas} épocas | {dev}", flush=True)
    X = cargar_imagenes(recs); t0 = time.time(); log = []
    firma_cfg = json.dumps({**cfg, "size": ENTRADA, "lrs": list(lrs), "epocas": epocas}, sort_keys=True)   # si cambia (p. ej. el backbone de main), la caché se recalcula
    for s, splits in enumerate(particiones(y, cows, a.division, a.semillas)):
        p = np.full(len(y), np.nan)
        for k, (tr, te) in enumerate(splits):
            if a.rapido and k > 0: break
            f = out / f"fold_s{s}_k{k}.npz"
            if f.exists() and "cfg" in np.load(f).files and str(np.load(f)["cfg"]) != firma_cfg:
                print(f"  [caché] {f.name} es de otra configuración: se recalcula", flush=True); f.unlink()
            if f.exists():
                z = np.load(f); p[te] = z["pred"]; log.append(dict(semilla=s, fold=k, lr=float(z["lr"]), epoca=int(z["epoca"]), mae_val=float(z["mae_val"]))); continue
            ia, iv = interna(tr, y, cows, a.division, s); best = None
            for lr in lrs:
                m, mae, ep, (mu, sd) = entrenar(cfg, X, y, ia, iv, lr, epocas, s, dev)
                print(f"  s{s} k{k} lr {lr:g}: MAE val {mae:.3f} (época {ep}) [{time.time() - t0:.0f}s]", flush=True)
                if best is None or mae < best[0]: best = (mae, lr, ep, predecir(m, X, te, dev) * sd + mu)
                del m; torch.cuda.empty_cache()
            np.savez(f, pred=best[3], lr=best[1], epoca=best[2], mae_val=best[0], cfg=firma_cfg); p[te] = best[3]
            log.append(dict(semilla=s, fold=k, lr=best[1], epoca=best[2], mae_val=round(best[0], 4)))
        ok = ~np.isnan(p)
        print(f"[semilla {s}] MAE test {np.abs(p[ok] - y[ok]).mean():.3f} ({ok.sum()} fotos) [{time.time() - t0:.0f}s]", flush=True)
        if not a.rapido: guardar_preds(out / f"preds_seed{s}.csv", names, cows, y, p)
        if a.rapido: break
    with open(out / "log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0])); w.writeheader(); w.writerows(log)
    print(f"-> {out} ({time.time() - t0:.0f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True, choices=list(MODELOS))
    ap.add_argument("--division", default="vaca", choices=DIVISIONES)
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--rapido", action="store_true", help="prueba de humo: 1 fold, 2 épocas, 1 lr")
    ap.add_argument("--data_dir", default="/data/Dataset_Fr_BCS_v2")
    ap.add_argument("--work_dir", default="/work")
    main(ap.parse_args())

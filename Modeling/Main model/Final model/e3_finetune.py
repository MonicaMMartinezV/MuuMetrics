# =============================================================
# Nombre del archivo: e3_finetune.py
#
# Descripción:
#   E3: DINOv2 con fine-tuning parcial (últimos K bloques) y la cabeza
#   que ganó con features congeladas: masked average pooling por región
#   (6 máscaras del segmentador) + capa lineal, pérdida MSE.
#   Mismos 5 folds por vaca que el resto de la comparación: cada fold
#   entrena con sus vacas "train", usa sus vacas "val" solo para early
#   stopping y predice sus vacas "test". Imagen completa aplastada a
#   size x size (igual que las features congeladas).
#
# Salida: <out_dir>/preds_<tag>.csv y reconstruye metrics/summary.
# =============================================================
import argparse, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision import tv_tensors

from filmModel import load_records
from backbones_frozen import mask_path, load_folds, MEAN, STD
from backbones_sweep import write_preds, rebuild_tables

class RegionDataset(Dataset):
    def __init__(self, records, masks_dir, size, train):
        self.records, self.masks_dir, self.size = records, masks_dir, size
        geo = [v2.RandomHorizontalFlip(), v2.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))] if train else []
        self.tf = v2.Compose(geo + [v2.ColorJitter(0.3, 0.3, 0.2, 0.02)]) if train else (lambda img, m: (img, m))

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]; s = self.size
        img = torch.from_numpy(np.asarray(Image.open(r["path"]).convert("RGB").resize((s, s), Image.BICUBIC))).permute(2, 0, 1)
        m = np.load(mask_path(r["path"], self.masks_dir))["masks"]
        m = F.interpolate(torch.from_numpy((m > 127).astype(np.uint8))[None].float(), size=(s, s), mode="nearest")[0].to(torch.uint8)
        img, m = self.tf(tv_tensors.Image(img), tv_tensors.Mask(m))   # misma geometría a imagen y máscaras
        x = (img.float() / 255.0 - torch.tensor(MEAN, dtype=torch.float32)[:, None, None]) / torch.tensor(STD, dtype=torch.float32)[:, None, None]
        return x, m.float(), torch.tensor(r["bcs"], dtype=torch.float32)

class RegionViT(nn.Module):
    def __init__(self, name, size, train_blocks, drop=0.3):
        super().__init__()
        self.vit = timm.create_model(name, pretrained=True, num_classes=0, img_size=size)
        for p in self.vit.parameters(): p.requires_grad = False
        n = len(self.vit.blocks)
        for b in self.vit.blocks[n - train_blocks:]: b.requires_grad_(True)
        self.vit.norm.requires_grad_(True)
        self.split = n - train_blocks
        C = self.vit.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(6 * C + 6, 1))

    def forward(self, x, masks):
        v = self.vit
        with torch.no_grad():                                   # bloques congelados sin grafo
            h = v.patch_drop(v.pos_drop(v._pos_embed(v.patch_embed(x)))) if hasattr(v, "patch_drop") else v.pos_drop(v._pos_embed(v.patch_embed(x)))
            h = v.norm_pre(h)
            for b in v.blocks[:self.split]: h = b(h)
        for b in v.blocks[self.split:]: h = b(h)
        h = v.norm(h)[:, v.num_prefix_tokens:]                   # [B, g*g, C]
        g = int(round(h.shape[1] ** 0.5))
        w = F.adaptive_avg_pool2d(masks, (g, g)).flatten(2)       # [B, 6, g*g]
        tot = w.sum(2, keepdim=True)
        vec = torch.bmm(w, h) / tot.clamp_min(1e-6)               # [B, 6, C]
        absent = (tot[..., 0] < 0.5).float()
        vec = vec * (1 - absent)[..., None]
        return self.head(torch.cat([vec.flatten(1), absent], 1))[:, 0]

def run_fold(a, recs_tr, recs_va, recs_te, dev, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ytr = np.array([r["bcs"] for r in recs_tr]); mu, sd = float(ytr.mean()), float(ytr.std() + 1e-6)
    mk = lambda recs, train: DataLoader(RegionDataset(recs, Path(a.masks_dir), a.size, train), batch_size=a.bs,
                                        shuffle=train, num_workers=a.workers, drop_last=train and len(recs) > a.bs)
    dl_tr, dl_va, dl_te = mk(recs_tr, True), mk(recs_va, False), mk(recs_te, False)
    model = RegionViT(a.model, a.size, a.train_blocks).to(dev)
    enc = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_enc}, {"params": model.head.parameters(), "lr": a.lr_head}], weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler()

    def predict(dl):
        model.eval(); out = []
        with torch.no_grad(), torch.autocast("cuda"):
            for x, m, _ in dl:
                x, m = x.to(dev), m.to(dev)
                p = (model(x, m) + model(x.flip(-1), m.flip(-1))) / 2   # TTA: volteo
                out.append(p.float().cpu().numpy())
        return np.concatenate(out) * sd + mu

    yva = np.array([r["bcs"] for r in recs_va]); best, best_state, bad = np.inf, None, 0
    for ep in range(a.epochs):
        model.train()
        for x, m, y in dl_tr:
            x, m, y = x.to(dev), m.to(dev), ((y - mu) / sd).to(dev)
            with torch.autocast("cuda"):
                loss = F.mse_loss(model(x, m).float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        sched.step()
        va = float(np.abs(predict(dl_va) - yva).mean())
        print(f"    epoch {ep+1}/{a.epochs} val_MAE={va:.3f}", flush=True)
        if va < best:
            best, bad = va, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k.startswith(("head.", "vit.norm.", "vit.blocks."))}
        else:
            bad += 1
            if bad >= a.patience: break
    model.load_state_dict(best_state, strict=False)
    return predict(dl_te), best

def main(a):
    t0 = time.time(); out = Path(a.out_dir); dev = torch.device("cuda")
    records = load_records(a.csv, a.images_dir, only_valid=True)
    y = np.array([r["bcs"] for r in records]); cows = np.array([r["vaca"] for r in records])
    folds = load_folds(Path(a.cv_dir), records)
    p = np.zeros(len(records)); log = []
    for k, (_, te) in enumerate(folds):
        s = json.load(open(Path(a.cv_dir) / "nofilm" / f"fold{k}" / "split.json"))["vacas"]
        rtr = [r for r in records if r["vaca"] in set(s["train"])]; rva = [r for r in records if r["vaca"] in set(s["val"])]
        preds = []
        for seed in range(a.seeds):
            pt, bva = run_fold(a, rtr, rva, [records[i] for i in te], dev, seed); preds.append(pt)
            print(f"[{a.tag}] fold {k} seed {seed}: best val MAE={bva:.3f} test MAE={np.abs(pt - y[te]).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)
        p[te] = np.mean(preds, 0); log.append({"fold": k, "test_mae": float(np.abs(p[te] - y[te]).mean())})
        torch.cuda.empty_cache()
    write_preds(out, a.tag, records, folds, cows, y, p)
    json.dump({"args": vars(a), "folds": log}, open(out / f"{a.tag}_log.json", "w"), indent=1)
    rebuild_tables(out, records, folds, y)
    print(f"E3 DONE {a.tag} MAE={np.abs(p - y).mean():.3f} ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset_frbcs_del.csv"); ap.add_argument("--images_dir", default="Dataset_Fr_BCS")
    ap.add_argument("--masks_dir", default="Dataset_Fr_BCS_masks"); ap.add_argument("--cv_dir", default="outputs_frbcs_cv")
    ap.add_argument("--out_dir", default="outputs_backbones_cv")
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m"); ap.add_argument("--size", type=int, default=378)
    ap.add_argument("--train_blocks", type=int, default=4)
    ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5); ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--tag", default="E3_dinov2L_regiones_ft")
    main(ap.parse_args())

#!/usr/bin/env python3
# finetune_convnext.py

# =============================================================
# Nombre del archivo: finetune_convnext.py
# Autores: Bárbara Paola Alcántara Vega
#           
#
# Descripción:
#Fine-tune ConvNeXt for regression (optimized for RTX 4090).

#Features:
# - timm convnext_base (fallback resnet50)
# - Mixed precision (AMP)
# - Optional MixUp and CutMix (for regression targets we mix 
#   numeric labels)
# - Strong augmentations, optional higher img_size (384/448/512)
# - Freeze backbone for initial epochs then unfreeze
# - OneCycleLR scheduler (or cosine) stepped per batch
# - Saves best_checkpoint_by_valMae.pth, preds_val.npz, 
#   preds_test.npz, history.json
#
# Dependencias: os, numpy, pathlib, typing, PIL, torch, 
#     
# =============================================================


import os, time, math, random, json, argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

# try timm
# Se intenta importar timm para ejecutar ConvNext-base
# Es importante contar con timm de antemano
# En caso de no poder emplear el ConvNext se emplea ResNet-50
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

# Funciones para 
def seedEverything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def DiscoverLabelMap(train_folder: Path):
    classes = sorted([p.name for p in train_folder.iterdir() if p.is_dir()])
    label_map = {}
    for c in classes:
        label_map[c] = float(c)
    return label_map

class FolderRegressionDataset(Dataset):
    def __init__(self, root: str, label_map: dict, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[str, float]] = []
        for cls_name, label_float in label_map.items():
            folder = self.root / cls_name
            if not folder.exists():
                continue
            for p in folder.iterdir():
                if p.suffix.lower() in [".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"]:
                    self.samples.append((str(p), float(label_float)))
        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {root}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32), path

# Construcción del modelo
def buildModel(name="convnext_base", pretrained=True):
    name = name.lower()
    if HAS_TIMM:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=1)
        except Exception as e:
            print(f"[timm] cannot create {name}: {e} -> fallback to resnet50")
    m = models.resnet50(pretrained=pretrained)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Linear(in_f,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,1))
    return m

# MixUp & CutMix para la regresión
def mixupRegression(x, y, alpha=0.2):
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y

def cutmixRegression(x, y, alpha=1.0):
    # x: [B,C,H,W], y: [B,1]
    if alpha <= 0:
        return x, y
    batch_size, _, H, W = x.size()
    lam = np.random.beta(alpha, alpha)
    # bounding box muestra
    rx = np.random.randint(W)
    ry = np.random.randint(H)
    rw = int(W * math.sqrt(1 - lam))
    rh = int(H * math.sqrt(1 - lam))
    x1 = np.clip(rx - rw // 2, 0, W)
    x2 = np.clip(rx + rw // 2, 0, W)
    y1 = np.clip(ry - rh // 2, 0, H)
    y2 = np.clip(ry + rh // 2, 0, H)
    idx = torch.randperm(batch_size).to(x.device)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    # Ajustar lambda al área verdadera de la img
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    mixed_y = lam_adj * y + (1.0 - lam_adj) * y[idx]
    return x, mixed_y

#  funciones de entrenamiento y evaluación
def trainOneEpoch(model, loader, optimizer, criterion, device, scaler, use_amp, scheduler, args):
    model.train()
    running_loss = 0.0
    preds = []; trues = []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.use_mixup:
            imgs, labels = mixupRegression(imgs, labels, alpha=args.mixup_alpha)
        if args.use_cutmix and (not args.use_mixup):
            imgs, labels = cutmixRegression(imgs, labels, alpha=args.cutmix_alpha)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss = loss / args.grad_accum_steps
        scaler.scale(loss).backward()
        if (0+1) % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        running_loss += loss.item() * imgs.size(0) * args.grad_accum_steps
        preds.extend(outputs.detach().cpu().numpy().reshape(-1).tolist())
        trues.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
    return running_loss / len(loader.dataset), preds, trues

def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    preds = []; trues = []; paths = []
    with torch.no_grad():
        for imgs, labels, pths in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.cpu().numpy().reshape(-1).tolist())
            trues.extend(labels.cpu().numpy().reshape(-1).tolist())
            paths.extend([str(x) for x in pths])
    return running_loss / len(loader.dataset), preds, trues, paths

# cálculo de métricas base 
# Se utilizan para ver el desempeño inicial del modelo durante la ejecución
# El resto de métricas se evalúa con un .py por separado para no cargar más el código
def compute_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

# MAIN
def main(args):
    seedEverything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    train_folder = data_dir / "train"
    if not train_folder.exists():
        raise FileNotFoundError(f"No encontré {train_folder}")

    label_map = DiscoverLabelMap(train_folder)
    print("Label map:", label_map)
    unique_labels = sorted(label_map.values())

    # transformaciones
    trainTf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.2,0.02)], p=0.7),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    valTf = transforms.Compose([
        transforms.Resize(int(args.img_size*1.05)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    trainDs = FolderRegressionDataset(str(data_dir/"train"), label_map, transform=trainTf)
    valDs   = FolderRegressionDataset(str(data_dir/"val"),   label_map, transform=valTf)
    testDs  = FolderRegressionDataset(str(data_dir/"test"),  label_map, transform=valTf)

    print("Sizes: train", len(trainDs), "val", len(valDs), "test", len(testDs))

    # Weighted sampler, no siempre se usa pero es muy útil si está desbalanceado
    labelsForWeights = [int((lab - min(unique_labels)) / 0.25) for _, lab in trainDs.samples]
    classCounts = np.bincount(labelsForWeights, minlength=len(unique_labels))
    classWeights = 1.0 / (classCounts + 1e-8)
    sampleWeights = [classWeights[i] for i in labelsForWeights]
    sampler = WeightedRandomSampler(sampleWeights, num_samples=len(sampleWeights), replacement=True) if args.use_sampler else None

    trainLoader = DataLoader(trainDs, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    valLoader = DataLoader(valDs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    testLoader = DataLoader(testDs, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    model = buildModel(args.model, pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    stepsPerEpoch = max(1, math.ceil(len(trainLoader)))
    totalSteps = args.epochs * stepsPerEpoch
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, totalSteps=totalSteps, pct_start=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type=="cuda")

    # freeze backbone para épocas iniciales
    if args.freeze_epochs > 0:
        print(f"Freezing non-head params for first {args.freeze_epochs} epochs")
        for n,p in model.named_parameters():
            if "head" in n or "classifier" in n or "fc" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    bestValMae = float("inf")
    bestCkpt = None
    history = {"trainMae": [], "valMae": []}

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        trainLoss, trainPreds, trainTrues = trainOneEpoch(model, trainLoader, optimizer, criterion, device, scaler, args.use_amp, scheduler, args)
        valLoss, valPreds, valTrues, valPaths = evaluate(model, valLoader, criterion, device, use_amp=args.use_amp)
        trainMae = float(np.mean(np.abs(np.array(trainPreds) - np.array(trainTrues))))
        valMae = float(np.mean(np.abs(np.array(valPreds) - np.array(valTrues))))
        history["trainMae"].append(trainMae); history["valMae"].append(valMae)

        print(f"Epoch {epoch}/{args.epochs}  trainMae={trainMae:.4f}  valMae={valMae:.4f}  time={(time.time()-t0):.1f}s")

        if epoch == args.freeze_epochs + 1:
            print("[UNFREEZE] Unfreezing all parameters.")
            for p in model.parameters(): p.requires_grad = True

        if valMae < bestValMae:
            bestValMae = valMae
            bestCkpt = Path(args.output_dir)/"best_checkpoint_by_valMae.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "label_map": label_map, "args": vars(args)}, str(bestCkpt))
            print("[SAVE] best checkpoint saved:", bestCkpt)

        # early stop:
        if epoch - (np.argmin(history["valMae"]) + 1) >= args.early_stop_patience:
            print("Early stopping patience reached.")
            break

    # guardado del historial
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir)/"history.json", "w") as f:
        json.dump(history, f, indent=2)

    # evaluar el mejor checkpoint en validación y prueba para salvar preds (con los paths)
    if bestCkpt is None:
        raise RuntimeError("No checkpoint saved (something wrong).")
    ckpt = torch.load(str(bestCkpt), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, valPreds, valTrues, valPaths = evaluate(model, valLoader, criterion, device, use_amp=args.use_amp)
    np.savez(Path(args.output_dir)/"preds_val.npz", preds=valPreds, labels=valTrues, paths=valPaths)

    test_loss, test_preds, test_trues, test_paths = evaluate(model, testLoader, criterion, device, use_amp=args.use_amp)
    np.savez(Path(args.output_dir)/"preds_test.npz", preds=test_preds, labels=test_trues, paths=test_paths)

    met = compute_metrics(test_trues, test_preds)
    print("Test metrics:", met)

    # guardar el modelo final (solo los pesos)
    torch.save({"model_state_dict": model.state_dict(), "label_map": label_map}, Path(args.output_dir)/"final_model.pth")
    print("Saved final model and preds in", args.output_dir)

# CLI 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="dataset")
    p.add_argument("--model", type=str, default="convnext_base")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="outputs_ft")
    p.add_argument("--use_amp", action="store_true", default=True)
    p.add_argument("--use_mixup", action="store_true", default=False)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--use_cutmix", action="store_true", default=False)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--use_sampler", action="store_true", default=False)
    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--scheduler", type=str, choices=["onecycle", "cosine"], default="onecycle")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
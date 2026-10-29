# =============================================================
# Nombre del archivo: filmModel.py
#
# Descripción:
#   Variante de finalModel.py que condiciona el ConvNeXt con los
#   días en leche (DEL) mediante FiLM (Feature-wise Linear
#   Modulation). Un MLP codifica el DEL y genera, para cada stage
#   seleccionado, un par (gamma, beta) por canal:
#
#       x_stage <- x_stage * (1 + dgamma(DEL)) + beta(DEL)
#
#   Las capas que producen (dgamma, beta) se inicializan en cero,
#   así que al inicio el modelo es idéntico al ConvNeXt
#   preentrenado y aprende a usar el DEL de forma gradual.
#
#   Las imágenes y etiquetas BCS salen del CSV (dataset_del.csv),
#   buscando cada ImgName dentro de --images_dir. El split
#   train/val/test se hace agrupado por vacaId (k-fold), para que
#   ninguna vaca aparezca en dos splits.
#
#   --no_film entrena el mismo pipeline sin DEL (baseline justo
#   con el mismo split).
#
# Dependencias:
#   torch, torchvision, timm, numpy, Pillow, sklearn
# =============================================================

import os, csv, time, math, json, argparse
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

from finalModel import seed_everything, compute_metrics

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Datos
def label_from_folder(path: Path):
    """BCS a partir del nombre de la carpeta padre ("3.25" -> 3.25); None si no es número."""
    try:
        return float(path.parent.name)
    except ValueError:
        return None

def load_records(csv_path: str, images_dir: str, only_valid: bool,
                 labels_from_folder: bool = False) -> List[Dict]:
    """
    Lee el CSV de DEL y resuelve la ruta de cada imagen.

    Args:
        csv_path (str): Ruta a dataset_del.csv.
        images_dir (str): Carpeta donde se buscan las imágenes (recursivo).
        only_valid (bool): Si True, descarta filas con valido=False.
        labels_from_folder (bool): Si True, el BCS sale de la carpeta de la
            imagen (<BCS>/img.jpg) en vez del CSV; las carpetas no numéricas
            (p.ej. "Discarted") se ignoran.

    Returns:
        list[dict]: Registros con path, bcs, del_days y vaca.
    """
    index = {}
    for p in Path(images_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if labels_from_folder and label_from_folder(p) is None:
                continue
            index.setdefault(p.stem.split("__")[-1], str(p))  # "ByIDxx__<foto>" -> "<foto>"

    records, missing = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if only_valid and row["valido"] != "True":
                continue
            path = index.get(row["ImgName"])
            if path is None:
                missing.append(row["ImgName"])
                continue
            records.append({
                "path": path,
                "bcs": label_from_folder(Path(path)) if labels_from_folder else float(row["BCS"]),
                "del_days": float(row["DEL"]),
                "vaca": row["vacaId"],
            })
    if missing:
        print(f"[WARN] {len(missing)} imágenes del CSV no encontradas en {images_dir} (ej: {missing[:3]})")
    if not records:
        raise RuntimeError("No se encontró ninguna imagen del CSV.")
    return records

def load_records_by_id(ids_dir: str, labels_dir: str) -> List[Dict]:
    """
    Registros a partir de una carpeta agrupada a mano por vaca
    (<ids_dir>/<vacaId>/img.jpg), sin usar el vacaId del CSV.
    El BCS se toma buscando el mismo nombre de archivo en
    <labels_dir>/<BCS>/img.jpg.
    - Solo cuentan las carpetas de primer nivel con nombre numérico
      (p.ej. "No encontre grupo" se ignora).
    - Los duplicados "<img> - Copy.jpg" se descartan si el original
      está en la misma vaca.
    - No hay DEL (del_days = 0): solo sirve con --no_film.
    """
    labels = {}
    for p in Path(labels_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS and label_from_folder(p) is not None:
            labels.setdefault(p.stem, label_from_folder(p))

    records, no_label, skipped = [], [], 0
    for cow_dir in sorted(d for d in Path(ids_dir).iterdir() if d.is_dir()):
        if not cow_dir.name.isdigit():
            skipped += sum(1 for p in cow_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
            continue
        seen = set()
        for p in sorted(cow_dir.iterdir()):
            if not (p.is_file() and p.suffix.lower() in IMG_EXTS):
                continue
            stem = p.stem.removesuffix(" - Copy")
            if stem in seen:
                skipped += 1
                continue
            seen.add(stem)
            if stem not in labels:
                no_label.append(f"{cow_dir.name}/{p.name}")
                continue
            records.append({"path": str(p), "bcs": labels[stem], "del_days": 0.0, "vaca": cow_dir.name})
    print(f"[by_id] {len(records)} imágenes, {len({r['vaca'] for r in records})} vacas | "
          f"ignoradas (sin grupo o duplicadas): {skipped} | sin BCS en {labels_dir}: {len(no_label)} {no_label[:3]}")
    if not records:
        raise RuntimeError(f"No se encontró ninguna imagen con BCS en {ids_dir}.")
    return records

MONTHS = {"ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6, "JULIO": 7,
          "AGOSTO": 8, "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12}

def load_records_cow_bcs(cow_bcs_dir: str) -> List[Dict]:
    """
    Registros a partir de carpetas <vacaId>_<BCS>/img.jpg (p.ej. Dataset_Fr_BCS),
    con vaca y BCS corregidos a mano en el nombre de la carpeta.
    - "<id>_MAYO4.25_JUNIO4_JULIO3.5": BCS por mes, según la fecha
      "...__AAAA-MM-..." del nombre de archivo.
    - Carpetas que no se pueden leer (p.ej. dos vacas mezcladas,
      "1552_Vaca1es2.5_Vaca2es4.75") se ignoran y se reportan.
    - No hay DEL (del_days = 0): solo sirve con --no_film.
    """
    records, skipped = [], []
    for cow_dir in sorted(d for d in Path(cow_bcs_dir).iterdir() if d.is_dir()):
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
            records.append({"path": str(p), "bcs": bcs, "del_days": 0.0, "vaca": cow})
    print(f"[cow_bcs] {len(records)} imágenes, {len({r['vaca'] for r in records})} vacas | ignoradas: {skipped}")
    if not records:
        raise RuntimeError(f"No se encontró ninguna imagen en {cow_bcs_dir}.")
    return records

def group_kfold_split(records: List[Dict], n_folds: int, fold: int, seed: int):
    """
    Reparte las vacas en n_folds grupos balanceados por número de
    imágenes. test = grupo `fold`, val = grupo `fold+1`, train = resto.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]} con vacaIds.
    """
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["vaca"]] = counts.get(r["vaca"], 0) + 1
    rng = np.random.RandomState(seed)
    cows = list(counts)
    rng.shuffle(cows)
    cows.sort(key=lambda c: -counts[c])  # sort estable: el shuffle desempata
    folds = [[] for _ in range(n_folds)]
    sizes = [0] * n_folds
    for c in cows:
        i = int(np.argmin(sizes))
        folds[i].append(c); sizes[i] += counts[c]
    test_f, val_f = fold % n_folds, (fold + 1) % n_folds
    return {
        "train": sorted(c for i, f in enumerate(folds) if i not in (test_f, val_f) for c in f),
        "val": sorted(folds[val_f]),
        "test": sorted(folds[test_f]),
    }

def holdout_split(records: List[Dict], ratios: List[float], by: str, seed: int) -> Dict[str, List[Dict]]:
    """
    Split único train/val/test con proporciones `ratios` (p.ej. 0.8,0.1,0.1).
    by="img": reparte imágenes al azar (split clásico; la misma vaca puede
    quedar en train y test). by="cow": reparte vacas, con las proporciones
    medidas en imágenes.

    Returns:
        dict: {"train": [...], "val": [...], "test": [...]} con registros.
    """
    rng = np.random.RandomState(seed)
    if by == "img":
        idx = rng.permutation(len(records))
        a, b = int(round(ratios[0] * len(idx))), int(round((ratios[0] + ratios[1]) * len(idx)))
        return {"train": [records[i] for i in idx[:a]], "val": [records[i] for i in idx[a:b]],
                "test": [records[i] for i in idx[b:]]}
    cows = sorted({r["vaca"] for r in records})
    rng.shuffle(cows)
    out, total, acc = {"train": [], "val": [], "test": []}, len(records), 0
    counts = {c: sum(r["vaca"] == c for r in records) for c in cows}
    for c in cows:
        k = "train" if acc < ratios[0] * total else "val" if acc < (ratios[0] + ratios[1]) * total else "test"
        out[k] += [r for r in records if r["vaca"] == c]
        acc += counts[c]
    return out

class DelRegressionDataset(Dataset):
    """
    Dataset de regresión que además devuelve el DEL normalizado.

    Args:
        records (list[dict]): Registros de load_records.
        del_scale (float): Divisor para normalizar DEL (p.ej. 305 = lactancia estándar).
        transform: Transformaciones de torchvision.
        target (str): "bcs" o "del" (DEL normalizado con del_scale).
    """
    def __init__(self, records: List[Dict], del_scale: float, transform=None, target: str = "bcs"):
        self.records = records
        self.del_scale = del_scale
        self.transform = transform
        self.target = target
        self.samples = [(r["path"], self._target(r)) for r in records]

    def _target(self, r):
        return r["bcs"] if self.target == "bcs" else r["del_days"] / self.del_scale

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        img = Image.open(r["path"]).convert("RGB")
        if self.transform: img = self.transform(img)
        del_norm = torch.tensor([r["del_days"] / self.del_scale], dtype=torch.float32)
        return img, del_norm, torch.tensor([self._target(r)], dtype=torch.float32), r["path"]

# Modelo
class DelEncoder(nn.Module):
    """
    Codifica el DEL normalizado con features de Fourier + MLP.
    Las Fourier features permiten capturar la forma no lineal de la
    curva de condición corporal a lo largo de la lactancia.
    """
    def __init__(self, n_freqs: int = 4, hidden: int = 64):
        super().__init__()
        self.register_buffer("freqs", (2.0 ** torch.arange(n_freqs)) * math.pi)
        self.mlp = nn.Sequential(
            nn.Linear(1 + 2 * n_freqs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.out_dim = hidden

    def forward(self, d):  # d: [B,1]
        ang = d * self.freqs
        return self.mlp(torch.cat([d, torch.sin(ang), torch.cos(ang)], dim=1))

class FiLM(nn.Module):
    """Genera (dgamma, beta) por canal y modula un mapa NCHW."""
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        dgamma, beta = self.proj(cond).float().chunk(2, dim=1)
        return x * (1 + dgamma[:, :, None, None].to(x.dtype)) + beta[:, :, None, None].to(x.dtype)

class ConvNeXtFiLM(nn.Module):
    """
    ConvNeXt de timm con FiLM a la salida de los stages indicados.

    Args:
        name (str): Modelo timm (familia convnext_*).
        pretrained (bool): Cargar pesos preentrenados.
        film_stages (list[int]): Stages (0-3) a modular. Vacío = sin FiLM.
        film_freqs (int): Frecuencias de Fourier del DEL (0 = solo DEL crudo).
        film_hidden (int): Ancho del MLP que codifica el DEL.
    """
    def __init__(self, name: str, pretrained: bool, film_stages: List[int],
                 film_freqs: int = 4, film_hidden: int = 64):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=1)
        if not hasattr(self.backbone, "stages"):
            raise ValueError(f"{name} no es un ConvNeXt de timm (sin .stages)")
        chs = [f["num_chs"] for f in self.backbone.feature_info]
        self.film_stages = list(film_stages)
        if self.film_stages:
            self.film_encoder = DelEncoder(n_freqs=film_freqs, hidden=film_hidden)
            self.film = nn.ModuleDict({
                str(i): FiLM(self.film_encoder.out_dim, chs[i]) for i in self.film_stages
            })

    def forward(self, x, del_norm):
        b = self.backbone
        cond = self.film_encoder(del_norm.float()) if self.film_stages else None
        x = b.stem(x)
        for i, stage in enumerate(b.stages):
            x = stage(x)
            if str(i) in getattr(self, "film", {}):
                x = self.film[str(i)](x, cond)
        x = b.norm_pre(x)
        return b.forward_head(x)

def is_trainable_when_frozen(name: str) -> bool:
    return "head" in name or "film" in name

# MixUp / CutMix (mezclan también el DEL con el mismo lambda)
def mix_batch(imgs, dels, labels, args):
    if args.use_mixup and args.mixup_alpha > 0:
        lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
    elif args.use_cutmix and args.cutmix_alpha > 0:
        _, _, H, W = imgs.shape
        lam0 = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
        rx, ry = np.random.randint(W), np.random.randint(H)
        rw, rh = int(W * math.sqrt(1 - lam0)), int(H * math.sqrt(1 - lam0))
        x1, x2 = np.clip(rx - rw // 2, 0, W), np.clip(rx + rw // 2, 0, W)
        y1, y2 = np.clip(ry - rh // 2, 0, H), np.clip(ry + rh // 2, 0, H)
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        imgs = imgs.clone()
        imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    else:
        return imgs, dels, labels
    return imgs, lam * dels + (1 - lam) * dels[idx], lam * labels + (1 - lam) * labels[idx]

# Entrenamiento y evaluación
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp, scheduler, args):
    model.train()
    running_loss = 0.0
    preds, trues = [], []
    optimizer.zero_grad()
    for step, (imgs, dels, labels, _) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        dels = dels.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        imgs, dels, labels = mix_batch(imgs, dels, labels, args)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(imgs, dels)
            loss = criterion(outputs.float(), labels)
        scaler.scale(loss / args.grad_accum_steps).backward()
        if step % args.grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if args.scheduler == "onecycle":
                scheduler.step()
        running_loss += loss.item() * imgs.size(0)
        preds.extend(outputs.detach().float().cpu().numpy().reshape(-1).tolist())
        trues.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
    if args.scheduler == "cosine":
        scheduler.step()
    return running_loss / len(loader.dataset), preds, trues

def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    running_loss = 0.0
    preds, trues, paths, dels_out = [], [], [], []
    with torch.no_grad():
        for imgs, dels, labels, pths in loader:
            imgs = imgs.to(device, non_blocking=True)
            dels = dels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(imgs, dels)
                loss = criterion(outputs.float(), labels)
            running_loss += loss.item() * imgs.size(0)
            preds.extend(outputs.float().cpu().numpy().reshape(-1).tolist())
            trues.extend(labels.cpu().numpy().reshape(-1).tolist())
            dels_out.extend(dels.cpu().numpy().reshape(-1).tolist())
            paths.extend([str(x) for x in pths])
    return running_loss / len(loader.dataset), preds, trues, paths, dels_out

# Función principal
def main(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_amp = args.use_amp and device.type == "cuda"

    if args.target == "del" and not args.no_film:
        print("[target=del] FiLM desactivado: el DEL es el objetivo, no puede ser entrada.")
        args.no_film = True
    if args.target == "del" and args.use_sampler:
        raise ValueError("--use_sampler balancea clases de BCS; no aplica con --target del.")
    # Las métricas se reportan en unidades del objetivo: BCS o días
    unit = args.del_scale if args.target == "del" else 1.0
    print("Objetivo:", "DEL (días)" if args.target == "del" else "BCS")
    film_stages = [] if args.no_film else [int(s) for s in args.film_stages.split(",") if s.strip()]
    print("FiLM stages:", film_stages or "ninguno (baseline sin DEL)")

    if args.cow_bcs_dir:
        if not args.no_film or args.target != "bcs":
            raise ValueError("--cow_bcs_dir no trae DEL: úsalo con --no_film y --target bcs.")
        records = load_records_cow_bcs(args.cow_bcs_dir)
        print(f"Vacas y BCS: nombres de carpeta de {args.cow_bcs_dir}")
    elif args.ids_dir:
        if not args.no_film or args.target != "bcs":
            raise ValueError("--ids_dir no trae DEL: úsalo con --no_film y --target bcs.")
        records = load_records_by_id(args.ids_dir, args.labels_dir)
        print(f"Vacas: carpetas de {args.ids_dir} | Etiquetas BCS: carpetas de {args.labels_dir}")
    else:
        records = load_records(args.csv, args.images_dir, args.only_valid, args.labels_from_folder)
        print("Etiquetas BCS:", "carpetas de --images_dir" if args.labels_from_folder else "CSV")
    if args.holdout:
        by_split = holdout_split(records, [float(x) for x in args.holdout.split(",")], args.holdout_by, args.split_seed)
        split = {k: sorted({r["vaca"] for r in v}) for k, v in by_split.items()}
        shared = len(set(split["test"]) & set(split["train"]))
        print(f"[holdout {args.holdout} por {args.holdout_by}] vacas de test que también están en train: {shared}")
    else:
        split = group_kfold_split(records, args.n_folds, args.fold, args.split_seed)
        if args.final:
            # Modelo final: el grupo de test se suma a train; val solo para early stopping
            split["train"], split["test"] = sorted(split["train"] + split["test"]), []
            print("[final] Sin test: train = todas las vacas menos las de val.")
        by_split = {k: [r for r in records if r["vaca"] in set(v)] for k, v in split.items()}
    with open(out_dir / "split.json", "w") as f:
        json.dump({"fold": args.fold, "n_folds": args.n_folds, "holdout": args.holdout, "holdout_by": args.holdout_by,
                   "vacas": split, "n_images": {k: len(v) for k, v in by_split.items()}}, f, indent=2)
    for k in ("train", "val", "test"):
        print(f"{k}: {len(by_split[k])} imágenes, {len(split[k])} vacas")

    # --square_resize: la foto completa se aplasta a cuadrado antes de todo, para
    # que el recorte aleatorio / central no deje fuera a la vaca en fotos 16:9
    square = [transforms.Resize((args.img_size, args.img_size))] if args.square_resize else []
    train_tf = transforms.Compose(square + [
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.2,0.02)], p=0.7),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_resize = square if args.square_resize else [
        transforms.Resize(int(args.img_size*1.05)),
        transforms.CenterCrop(args.img_size),
    ]
    val_tf = transforms.Compose(val_resize + [
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = DelRegressionDataset(by_split["train"], args.del_scale, train_tf, args.target)
    val_ds = DelRegressionDataset(by_split["val"], args.del_scale, val_tf, args.target)
    test_ds = DelRegressionDataset(by_split["test"], args.del_scale, val_tf, args.target)

    sampler = None
    if args.use_sampler:
        min_label = min(lab for _, lab in train_ds.samples)
        bins = [int(round((lab - min_label) / 0.25)) for _, lab in train_ds.samples]
        counts = np.bincount(bins)
        sampler = WeightedRandomSampler([1.0 / counts[b] for b in bins], num_samples=len(bins), replacement=True)

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
                     persistent_workers=(args.num_workers > 0))
    train_loader = DataLoader(train_ds, shuffle=(sampler is None), sampler=sampler, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = ConvNeXtFiLM(args.model, pretrained=True, film_stages=film_stages,
                         film_freqs=args.film_freqs, film_hidden=args.film_hidden).to(device)

    criterion = nn.SmoothL1Loss()
    film_params = [p for n, p in model.named_parameters() if "film" in n]
    base_params = [p for n, p in model.named_parameters() if "film" not in n]
    groups = [{"params": base_params, "lr": args.lr}]
    if film_params:
        groups.append({"params": film_params, "lr": args.lr * args.film_lr_mult})
    optimizer = optim.AdamW(groups, weight_decay=args.weight_decay)

    opt_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[g["lr"] for g in groups],
                                                  total_steps=args.epochs * opt_steps_per_epoch, pct_start=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if args.freeze_epochs > 0:
        print(f"Freezing backbone (excepto head y FiLM) por {args.freeze_epochs} épocas")
        for n, p in model.named_parameters():
            p.requires_grad = is_trainable_when_frozen(n)

    select_key = f"val_{args.select_by}"
    best_val_metric = float("inf")
    best_ckpt = out_dir / f"best_checkpoint_by_{select_key}.pth"
    saved = False
    history = {"train_mae": [], "val_mae": [], "train_rmse": [], "val_rmse": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        _, train_preds, train_trues = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                                      scaler, use_amp, scheduler, args)
        _, val_preds, val_trues, _, _ = evaluate(model, val_loader, criterion, device, use_amp)
        tp, tt = np.array(train_preds) * unit, np.array(train_trues) * unit
        vp, vt = np.array(val_preds) * unit, np.array(val_trues) * unit
        train_mae, val_mae = float(np.mean(np.abs(tp - tt))), float(np.mean(np.abs(vp - vt)))
        train_rmse, val_rmse = float(np.sqrt(np.mean((tp - tt) ** 2))), float(np.sqrt(np.mean((vp - vt) ** 2)))
        history["train_mae"].append(train_mae); history["val_mae"].append(val_mae)
        history["train_rmse"].append(train_rmse); history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch}/{args.epochs}  train_MAE={train_mae:.4f}  val_MAE={val_mae:.4f}  "
              f"train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}  time={(time.time()-t0):.1f}s")

        if epoch == args.freeze_epochs:
            print("[UNFREEZE] Unfreezing all parameters.")
            for p in model.parameters(): p.requires_grad = True

        val_metric = val_mae if args.select_by == "mae" else val_rmse
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "args": vars(args)}, str(best_ckpt))
            saved = True
            print("[SAVE] best checkpoint saved:", best_ckpt)

        if epoch - (int(np.argmin(history[select_key])) + 1) >= args.early_stop_patience:
            print("Early stopping patience reached.")
            break

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    if not saved:
        raise RuntimeError("No checkpoint saved (something wrong).")

    model.load_state_dict(torch.load(str(best_ckpt), map_location=device)["model_state"])
    _, val_preds, val_trues, val_paths, val_dels = evaluate(model, val_loader, criterion, device, use_amp)
    val_preds, val_trues = (np.array(val_preds) * unit).tolist(), (np.array(val_trues) * unit).tolist()
    np.savez(out_dir / "preds_val.npz", preds=val_preds, labels=val_trues, paths=val_paths,
             del_days=np.array(val_dels) * args.del_scale)
    val_met = compute_metrics(val_trues, val_preds)
    print("Val metrics:", val_met)
    with open(out_dir / "val_metrics.json", "w") as f:
        json.dump(val_met, f, indent=2)

    if not args.final:
        _, test_preds, test_trues, test_paths, test_dels = evaluate(model, test_loader, criterion, device, use_amp)
        test_preds, test_trues = (np.array(test_preds) * unit).tolist(), (np.array(test_trues) * unit).tolist()
        np.savez(out_dir / "preds_test.npz", preds=test_preds, labels=test_trues, paths=test_paths,
                 del_days=np.array(test_dels) * args.del_scale)
        save_test_metrics(test_trues, test_preds, train_ds, unit, out_dir)

    torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, out_dir / "final_model.pth")
    print("Saved final model and preds in", out_dir)

def save_test_metrics(test_trues, test_preds, train_ds, unit, out_dir):
    met = compute_metrics(test_trues, test_preds)
    # Referencia trivial: predecir siempre la media de train (sin mirar la foto)
    train_mean = float(np.mean([lab for _, lab in train_ds.samples])) * unit
    met["mae_media_train"] = float(np.mean(np.abs(np.array(test_trues) - train_mean)))
    print("Test metrics:", met)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(met, f, indent=2)

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, default="dataset_del.csv")
    p.add_argument("--images_dir", type=str, default="Images",
                   help="Carpeta con las imágenes (se busca ImgName recursivamente).")
    p.add_argument("--labels_from_folder", action="store_true", default=False,
                   help="Tomar el BCS de la carpeta de cada imagen (<BCS>/img.jpg) en vez del CSV.")
    p.add_argument("--ids_dir", type=str, default=None,
                   help="Carpeta <vacaId>/img.jpg agrupada a mano (p.ej. Dataset/ByID): imágenes y vaca de aquí, "
                        "ignorando el CSV. Requiere --no_film.")
    p.add_argument("--labels_dir", type=str, default="Images",
                   help="Con --ids_dir: carpeta <BCS>/img.jpg de donde sale el BCS (por nombre de archivo).")
    p.add_argument("--cow_bcs_dir", type=str, default=None,
                   help="Carpeta <vacaId>_<BCS>/img.jpg (p.ej. Dataset/Dataset_Fr_BCS): vaca y BCS del nombre de "
                        "la carpeta, ignorando el CSV. Requiere --no_film.")
    p.add_argument("--square_resize", action="store_true", default=False,
                   help="Reescalar la imagen completa a cuadrado (sin recorte central); para fotos sin recortar.")
    p.add_argument("--only_valid", action="store_true", default=False,
                   help="Usar solo filas con valido=True.")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0, help="Fold de test; val = fold+1.")
    p.add_argument("--final", action="store_true", default=False,
                   help="Modelo final: sin test, train = todas las vacas salvo el grupo de val (fold+1).")
    p.add_argument("--holdout", type=str, default=None,
                   help="Split único train,val,test (p.ej. 0.8,0.1,0.1) en vez del k-fold por vaca.")
    p.add_argument("--holdout_by", type=str, choices=["img", "cow"], default="cow",
                   help="Con --holdout: repartir imágenes al azar (img, split clásico) o vacas (cow).")
    p.add_argument("--split_seed", type=int, default=42,
                   help="Semilla del reparto de vacas en folds (independiente de --seed).")
    p.add_argument("--target", type=str, choices=["bcs", "del"], default="bcs",
                   help="Qué predecir desde la foto: BCS o DEL (días; desactiva FiLM).")
    p.add_argument("--no_film", action="store_true", default=False,
                   help="Baseline sin DEL con el mismo split/pipeline.")
    p.add_argument("--film_stages", type=str, default="1,2,3",
                   help="Stages de ConvNeXt (0-3) a modular con FiLM, separados por coma.")
    p.add_argument("--film_lr_mult", type=float, default=10.0)
    p.add_argument("--film_freqs", type=int, default=4,
                   help="Frecuencias de Fourier para codificar el DEL (0 = solo DEL crudo).")
    p.add_argument("--film_hidden", type=int, default=64, help="Ancho del MLP del DEL.")
    p.add_argument("--del_scale", type=float, default=305.0,
                   help="DEL se divide entre este valor (305 = lactancia estándar).")
    p.add_argument("--model", type=str, default="convnext_base")
    p.add_argument("--img_size", type=int, default=384)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="outputs_film")
    p.add_argument("--use_amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_mixup", action="store_true", default=False)
    p.add_argument("--mixup_alpha", type=float, default=0.2)
    p.add_argument("--use_cutmix", action="store_true", default=False)
    p.add_argument("--cutmix_alpha", type=float, default=1.0)
    p.add_argument("--use_sampler", action="store_true", default=False)
    p.add_argument("--freeze_epochs", type=int, default=3)
    p.add_argument("--scheduler", type=str, choices=["onecycle", "cosine"], default="onecycle")
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--early_stop_patience", type=int, default=6)
    p.add_argument("--select_by", type=str, choices=["mae", "rmse"], default="mae")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main(parse_args())

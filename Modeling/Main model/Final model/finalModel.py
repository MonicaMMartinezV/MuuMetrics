# =============================================================
# Nombre del archivo: finalModel.py
# Autor: Bárbara Paola Alcántara Vega 
#
# Descripción:
#   Script para fine-tuning del modelo ConvNeXt en una tarea de
#   regresión. Incluye soporte para AMP, MixUp, CutMix, congelamiento
#   progresivo del backbone y scheduler OneCycleLR.
#
# Dependencias:
#   torch, torchvision, timm, numpy, Pillow, sklearn
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

# Intentar importar timm
try:
    import timm
    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

# Funciones utilitarias
def seed_everything(seed: int = 42):
    """
    Fija todas las semillas para garantizar reproducibilidad.

    Args:
        seed (int): Valor entero de semilla aleatoria.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discover_label_map(train_folder: Path):
    """
    Genera un mapeo entre nombres de carpetas y etiquetas numéricas.

    Args:
        train_folder (Path): Ruta a la carpeta raíz que contiene subcarpetas
            nombradas según el valor de la etiqueta (por ejemplo: "2.5", "3.0").

    Returns:
        dict: Diccionario {nombre_carpeta: valor_float}.
    """
    classes = sorted([p.name for p in train_folder.iterdir() if p.is_dir()])
    label_map = {}
    for c in classes:
        label_map[c] = float(c)
    return label_map

# Dataset personalizado para regresión
class FolderRegressionDataset(Dataset):
    """
    Dataset para regresión basado en carpetas donde cada subcarpeta
    representa una etiqueta numérica.

    Args:
        root (str): Ruta a la carpeta principal (train/val/test).
        label_map (dict): Mapeo de nombres de carpeta a etiquetas numéricas.
        transform (torchvision.transforms.Compose, opcional):
            Transformaciones de preprocesamiento.

    Raises:
        RuntimeError: Si no se encuentran imágenes en la ruta dada.
    """
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
    """
    Devuelve el número total de muestras en el dataset.
    Returns:
        int: Número de imágenes disponibles.
    """
    def __getitem__(self, idx):
        """
        Obtiene una imagen y su etiqueta asociada.

        Args:
            idx (int): Índice de la muestra.

        Returns:
            tuple:
                - img (torch.Tensor): Imagen procesada (C,H,W).
                - label (torch.Tensor): Valor numérico de la etiqueta.
                - path (str): Ruta absoluta de la imagen.
        """
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor([label], dtype=torch.float32), path

# Construcción del modelo
def build_model(name="convnext_base", pretrained=True):
    """
    Crea un modelo de regresión basado en ConvNeXt o ResNet50.

    Args:
        name (str): Nombre del modelo base (por defecto: "convnext_base").
        pretrained (bool): Indica si se cargan pesos preentrenados.

    Returns:
        nn.Module: Modelo de PyTorch configurado para regresión.
    """
    name = name.lower()
    if HAS_TIMM:
        try:
            return timm.create_model(name, pretrained=pretrained, num_classes=1)
        except Exception as e:
            print(f"[timm] cannot create {name}: {e} -> fallback to resnet50")
    m = models.resnet50(pretrained=pretrained)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(
        nn.Linear(in_f,512), 
        nn.ReLU(), 
        nn.Dropout(0.3), 
        nn.Linear(512,1)
    )
    return m

# MixUp y CutMix adaptados a regresión
def mixup_regression(x, y, alpha=0.2):
    """
    Aplica la técnica MixUp sobre imágenes y etiquetas continuas.

    Args:
        x (torch.Tensor): Batch de imágenes [B, C, H, W].
        y (torch.Tensor): Batch de etiquetas [B, 1].
        alpha (float): Parámetro beta para la mezcla.

    Returns:
        tuple:
            - mixed_x (torch.Tensor): Imágenes combinadas.
            - mixed_y (torch.Tensor): Etiquetas combinadas.
    """
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    return mixed_x, mixed_y

def cutmix_regression(x, y, alpha=1.0):
    """
    Aplica la técnica CutMix en tareas de regresión.

    Args:
        x (torch.Tensor): Batch de imágenes [B, C, H, W].
        y (torch.Tensor): Batch de etiquetas [B, 1].
        alpha (float): Parámetro beta para el tamaño de recorte.

    Returns:
        tuple:
            - mixed_x (torch.Tensor): Imágenes modificadas.
            - mixed_y (torch.Tensor): Etiquetas ajustadas.
    """
    if alpha <= 0:
        return x, y
    batch_size, _, H, W = x.size()
    lam = np.random.beta(alpha, alpha)
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
    lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
    mixed_y = lam_adj * y + (1.0 - lam_adj) * y[idx]
    return x, mixed_y

# Entrenamiento y evaluación
def train_one_epoch(
        model, 
        loader, 
        optimizer, 
        criterion, 
        device, 
        scaler, 
        use_amp, 
        scheduler, 
        args
    ):
    """
    Ejecuta una época completa de entrenamiento.

    Args:
        model (nn.Module): Modelo a entrenar.
        loader (DataLoader): Cargador de datos de entrenamiento.
        optimizer (Optimizer): Optimizador de PyTorch.
        criterion (nn.Module): Función de pérdida.
        device (torch.device): Dispositivo de ejecución.
        scaler (GradScaler): Escalador para AMP.
        use_amp (bool): Si se usa precisión mixta.
        scheduler: Planificador de tasa de aprendizaje.
        args (Namespace): Parámetros del CLI.

    Returns:
        tuple:
            - avg_loss (float): Pérdida promedio por muestra.
            - preds (list): Predicciones del modelo.
            - trues (list): Valores reales.
    """
    model.train()
    running_loss = 0.0
    preds = []; trues = []
    optimizer.zero_grad()
    for step, (imgs, labels, _) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.use_mixup:
            imgs, labels = mixup_regression(imgs, labels, alpha=args.mixup_alpha)
        if args.use_cutmix and (not args.use_mixup):
            imgs, labels = cutmix_regression(imgs, labels, alpha=args.cutmix_alpha)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss = loss / args.grad_accum_steps
        scaler.scale(loss).backward()
        # Paso del optimizador cada grad_accum_steps batches (y en el último
        # batch de la época, para no perder gradientes acumulados)
        if step % args.grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            # OneCycleLR avanza por paso de optimizador; Cosine se avanza por época en main()
            if args.scheduler == "onecycle":
                scheduler.step()
        running_loss += loss.item() * imgs.size(0) * args.grad_accum_steps
        preds.extend(outputs.detach().cpu().numpy().reshape(-1).tolist())
        trues.extend(labels.detach().cpu().numpy().reshape(-1).tolist())
    return running_loss / len(loader.dataset), preds, trues

def evaluate(model, loader, criterion, device, use_amp):
    """
    Evalúa el modelo en un conjunto de validación o prueba.

    Args:
        model (nn.Module): Modelo entrenado.
        loader (DataLoader): Cargador de datos.
        criterion (nn.Module): Función de pérdida.
        device (torch.device): Dispositivo (CPU o GPU).
        use_amp (bool): Si se usa AMP.

    Returns:
        tuple:
            - avg_loss (float): Pérdida promedio.
            - preds (list): Predicciones.
            - trues (list): Etiquetas verdaderas.
            - paths (list): Rutas de imágenes evaluadas.
    """
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

# Métricas
def compute_metrics(y_true, y_pred):
    """
    Calcula métricas de evaluación de regresión.

    Args:
        y_true (list): Valores reales.
        y_pred (list): Predicciones del modelo.

    Returns:
        dict: Contiene MAE, RMSE y R².
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}

# Función principal
def main(args):
    """
    Ejecuta el proceso completo de entrenamiento, validación y prueba.

    Args:
        args (Namespace): Argumentos obtenidos del CLI.

    Returns:
        None
    """
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.backends.cudnn.benchmark = True

    data_dir = Path(args.data_dir)
    train_folder = data_dir / "train"
    if not train_folder.exists():
        raise FileNotFoundError(f"No encontré {train_folder}")

    label_map = discover_label_map(train_folder)
    print("Label map:", label_map)
    unique_labels = sorted(label_map.values())

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.3,0.3,0.2,0.02)], p=0.7),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size*1.05)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = FolderRegressionDataset(str(data_dir/"train"), label_map, transform=train_tf)
    val_ds   = FolderRegressionDataset(str(data_dir/"val"),   label_map, transform=val_tf)
    test_ds  = FolderRegressionDataset(str(data_dir/"test"),  label_map, transform=val_tf)

    print("Sizes: train", len(train_ds), "val", len(val_ds), "test", len(test_ds))

    labels_for_weights = [int((lab - min(unique_labels)) / 0.25) for _, lab in train_ds.samples]
    class_counts = np.bincount(labels_for_weights, minlength=len(unique_labels))
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[i] for i in labels_for_weights]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True) if args.use_sampler else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers>0))

    model = build_model(args.model, pretrained=True).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = args.epochs * steps_per_epoch
    if args.scheduler == "onecycle":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and device.type=="cuda")

    if args.freeze_epochs > 0:
        print(f"Freezing non-head params for first {args.freeze_epochs} epochs")
        for n,p in model.named_parameters():
            if "head" in n or "classifier" in n or "fc" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    select_key = f"val_{args.select_by}"  # "val_mae" o "val_rmse", según --select_by
    best_val_metric = float("inf")
    best_ckpt = None
    history = {"train_mae": [], "val_mae": [], "train_rmse": [], "val_rmse": []}

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        train_loss, train_preds, train_trues = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, args.use_amp, scheduler, args)
        if args.scheduler == "cosine":
            scheduler.step()
        val_loss, val_preds, val_trues, val_paths = evaluate(model, val_loader, criterion, device, use_amp=args.use_amp)
        train_preds_arr, train_trues_arr = np.array(train_preds), np.array(train_trues)
        val_preds_arr, val_trues_arr = np.array(val_preds), np.array(val_trues)
        train_mae = float(np.mean(np.abs(train_preds_arr - train_trues_arr)))
        val_mae = float(np.mean(np.abs(val_preds_arr - val_trues_arr)))
        train_rmse = float(np.sqrt(np.mean((train_preds_arr - train_trues_arr) ** 2)))
        val_rmse = float(np.sqrt(np.mean((val_preds_arr - val_trues_arr) ** 2)))
        history["train_mae"].append(train_mae); history["val_mae"].append(val_mae)
        history["train_rmse"].append(train_rmse); history["val_rmse"].append(val_rmse)

        print(f"Epoch {epoch}/{args.epochs}  train_MAE={train_mae:.4f}  val_MAE={val_mae:.4f}  "
              f"train_RMSE={train_rmse:.4f}  val_RMSE={val_rmse:.4f}  time={(time.time()-t0):.1f}s")

        if epoch == args.freeze_epochs + 1:
            print("[UNFREEZE] Unfreezing all parameters.")
            for p in model.parameters(): p.requires_grad = True

        val_metric = val_mae if args.select_by == "mae" else val_rmse
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            best_ckpt = Path(args.output_dir)/f"best_checkpoint_by_{select_key}.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "label_map": label_map, "args": vars(args)}, str(best_ckpt))
            print("[SAVE] best checkpoint saved:", best_ckpt)

        if epoch - (np.argmin(history[select_key]) + 1) >= args.early_stop_patience:
            print("Early stopping patience reached.")
            break

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir)/"history.json", "w") as f:
        json.dump(history, f, indent=2)

    if best_ckpt is None:
        raise RuntimeError("No checkpoint saved (something wrong).")
    ckpt = torch.load(str(best_ckpt), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, val_preds, val_trues, val_paths = evaluate(model, val_loader, criterion, device, use_amp=args.use_amp)
    np.savez(Path(args.output_dir)/"preds_val.npz", preds=val_preds, labels=val_trues, paths=val_paths)

    test_loss, test_preds, test_trues, test_paths = evaluate(model, test_loader, criterion, device, use_amp=args.use_amp)
    np.savez(Path(args.output_dir)/"preds_test.npz", preds=test_preds, labels=test_trues, paths=test_paths)

    met = compute_metrics(test_trues, test_preds)
    print("Test metrics:", met)

    torch.save({"model_state_dict": model.state_dict(), "label_map": label_map}, Path(args.output_dir)/"final_model.pth")
    print("Saved final model and preds in", args.output_dir)
# CLI (Command Line Interface)
def parse_args():
    """
    Define y parsea los argumentos de línea de comando.

    Returns:
        argparse.Namespace: Objeto con los parámetros configurados.
    """
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
    p.add_argument("--select_by", type=str, choices=["mae", "rmse"], default="mae",
                   help="Métrica de validación usada para elegir el mejor checkpoint y el early stopping.")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

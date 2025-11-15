#!/usr/bin/env python3

# =============================================================
# Nombre del archivo: split_dataset.py
# Autores: Bárbara Paola Alcántara Vega
#           
# Descripción:
# divide el dataset en train/validation/test 
# Soporta:
# - ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
# Salida: dataset dividido en train/val/test (80/10/10)
#
# Dependencias: os, shutil, pathlib, sklearn.model_selection,
#               argparse
#     
# =============================================================

import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Las siguientes funcines se utilizan para extraer las imágenes y sus clasificaciones
# Cada carpeta contiene la clasificación y dentro de las carpetas las imágenes
# con sus nombres intactos. Se les debe asignar la calificación a la que pertenecen en
# las carpetas.
def isImageFile(p: Path):
    return p.suffix.lower() in VALID_EXTS and p.is_file()

def gatherClassFiles(class_path: Path):
    return sorted([p for p in class_path.iterdir() if isImageFile(p)])

def safePlace(file_path: Path, dst_dir: Path, copy=True):
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / file_path.name
    try:
        if copy:
            shutil.copy2(file_path, dst)
        else:
            shutil.move(str(file_path), dst)
    except Exception as e:
        print(f"[ERROR] Al copiar/mover {file_path} -> {dst}: {e}")

def main(raw_dir, out_dir, train_ratio, val_ratio, test_ratio, copy_files, random_state):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Las proporciones deben sumar 1.0"
    raw = Path(raw_dir)
    if not raw.exists():
        print(f"[ERROR] No existe {raw.resolve()}")
        return

    out = Path(out_dir)
    for split in ("train","val","test"):
        (out / split).mkdir(parents=True, exist_ok=True)

    classes = [p for p in raw.iterdir() if p.is_dir()]
    if not classes:
        print(f"[ERROR] No se encontraron subcarpetas en {raw.resolve()}")
        return

    print("Clases detectadas:", [c.name for c in classes])

    for cls_path in classes:
        cls_name = cls_path.name
        files = gatherClassFiles(cls_path)
        n = len(files)
        if n == 0:
            print(f"[WARN] Clase '{cls_name}' sin imágenes válidas, se omite.")
            continue
        if n == 1:
            # todo a train (evitamos crear val/test vacíos)
            train_files = files
            val_files = []
            test_files = []
        else:
            train_files, temp_files = train_test_split(
                files, train_size=train_ratio, random_state=random_state, shuffle=True
            )
            # Manejar temp_files con 0,1 elemento
            if len(temp_files) == 0:
                val_files, test_files = [], []
            elif len(temp_files) == 1:
                val_files, test_files = temp_files, []
            else:
                rel = val_ratio / (val_ratio + test_ratio)
                val_files, test_files = train_test_split(
                    temp_files, train_size=rel, random_state=random_state, shuffle=True
                )

        # Colocar archivos
        for f in train_files:
            safePlace(f, out / "train" / cls_name, copy_files)
        for f in val_files:
            safePlace(f, out / "val" / cls_name, copy_files)
        for f in test_files:
            safePlace(f, out / "test" / cls_name, copy_files)

        print(f"{cls_name}: total={n} -> train={len(train_files)} val={len(val_files)} test={len(test_files)}")

    print("Split completado en:", out.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", default="DS")
    parser.add_argument("--out_dir", default="dataset")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--copy", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.raw_dir, args.out_dir, args.train, args.val, args.test, args.copy, args.seed)
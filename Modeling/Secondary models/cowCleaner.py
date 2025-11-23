# =============================================================
# Nombre del archivo: cowCleaner.py
# Autor: Grant Nathaniel Keegan
# Fecha de creación: 10-20-2025
# Descripción: Archivo para limpiar los datos de imágenes del
# proyecto MuuMetrics. Inteligencia Artificial Para la Ciencia
# de Datos II.
# Dependencias: os, time, shitul, pathlib, tqdm, ultralytics, PIL
# =============================================================

# Dependencias generales.
import os
import time
import shutil
from pathlib import Path
from tqdm import tqdm
# Dependencias "Separar Oscuras"
import numpy as np, cv2
# Dependencias "Separar Vacas"
from ultralytics import YOLO
# Dependencias "Aplicar Brillo"
from PIL import Image, ImageEnhance

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp",
              ".JPG", ".JPEG", ".PNG", ".BMP", ".TIF", ".TIFF", ".WEBP"}

# ============ Utilidades comunes ============

def safeMove(src: Path, dst_dir: Path):
    """
    Mueve con manejo de colisiones y fallback copy2->remove (útil en OneDrive).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        stem, suf = dst.stem, dst.suffix
        dst = dst_dir / f"{stem}_{int(time.time()*1000)}{suf}"
    try:
        shutil.move(str(src), str(dst))
        return True
    except Exception:
        try:
            shutil.copy2(str(src), str(dst))
            os.remove(str(src))
            return True
        except Exception as e:
            print(f"  ⚠️  No se pudo mover/copiar {src.name}: {e}")
            return False

# ============ 1) Separar oscuras / no oscuras ============

def cv2ReadWin(pathSTR: str):
    """
    Lectura robusta para Windows/OneDrive/rutas largas:
    """
    p = pathSTR
    if os.name == "nt" and not p.startswith("\\\\?\\") and len(p) > 240:
        p = "\\\\?\\" + os.path.abspath(p)
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None

def isDarkImage(img, brightnessThreshold: float, dark_percent: float) -> bool:
    import numpy as np, cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avg_brightness = float(np.mean(gray))
    pct_dark = float((gray < brightnessThreshold).mean())
    return (avg_brightness < brightnessThreshold) and (pct_dark > dark_percent)

def separarateDark():
    """
    Mueve imágenes 'oscuras' a una carpeta y las 'no oscuras' a otra.
    Criterio: % de píxeles con nivel de gris < umbral.
    """
    import cv2  # solo se requiere si ejecutas esta opción
    import numpy as np

    print("\n=== 1) Separar imágenes oscuras ===")
    INPUT_DIR = input("📂 Carpeta de entrada: ").strip()
    OUTPUT_DIR = input("📂 Carpeta destino IMÁGENES NO OSCURAS: ").strip()
    DARK_DIR   = input("📂 Carpeta destino IMÁGENES OSCURAS: ").strip()

    try:
        BRIGHTNESS_THRESHOLD = float(input("Umbral brillo [0-255] (def=40): ") or "40")
        DARK_PERCENT = float(input("Proporción mínima de píxeles oscuros [0-1] (def=0.80): ") or "0.80")
    except ValueError:
        print("Valores inválidos; usando 40 y 0.80.")
        BRIGHTNESS_THRESHOLD, DARK_PERCENT = 40.0, 0.80

    inPath  = Path(INPUT_DIR).resolve()
    outPath = Path(OUTPUT_DIR).resolve()
    dark_path= Path(DARK_DIR).resolve()

    if not inPath.is_dir():
        print(f"❌ La carpeta de entrada no existe: {inPath}")
        return

    outPath.mkdir(parents=True, exist_ok=True)
    dark_path.mkdir(parents=True, exist_ok=True)

    files = [p for p in inPath.iterdir() if p.suffix in VALID_EXTS and p.is_file()]
    if not files:
        print("⚠️ No se encontraron imágenes soportadas.")
        return

    moved_dark = moved_light = skipped = 0

    for src in tqdm(files, desc="Procesando (oscuras)"):
        img = cv2ReadWin(str(src))
        if img is None:
            skipped += 1
            continue

        if isDarkImage(img, BRIGHTNESS_THRESHOLD, DARK_PERCENT):
            ok = safeMove(src, dark_path)
            moved_dark += int(ok)
            skipped += int(not ok)
        else:
            ok = safeMove(src, outPath)
            moved_light += int(ok)
            skipped += int(not ok)

    print("\n--- Resumen (oscuras) ---")
    print(f"Total encontrados   : {len(files)}")
    print(f"Movidos a OSCURAS   : {moved_dark}")
    print(f"Movidos a NO OSCURAS: {moved_light}")
    print(f"Saltados/errores    : {skipped}")
    print(f"Parámetros          : brillo<{BRIGHTNESS_THRESHOLD}, %oscuro>{DARK_PERCENT}")

# ============ 2) Separar con vaca / sin vaca (YOLOv8) ============

def separarateCows():
    """
    Usa YOLOv8 (Ultralytics) preentrenado en COCO (incluye 'cow').
    Si detecta ≥1 'cow' mueve a carpeta CON vaca; en caso contrario a SIN vaca.
    """

    print("\n=== 2) Separar imágenes con/sin vacas (YOLOv8) ===")
    INPUT_DIR = input("📂 Carpeta de entrada: ").strip()
    COW_DIR   = input("📂 Carpeta destino CON vacas: ").strip()
    NOCOW_DIR = input("📂 Carpeta destino SIN vacas (enter para omitir moverlas): ").strip()

    device = (input("Dispositivo [cpu / 0 / 1 ...] (def=cpu): ").strip() or "cpu")
    try:
        conf = float(input("Confianza mínima (def=0.10): ") or "0.10")
        iou  = float(input("IOU (def=0.40): ") or "0.40")
    except ValueError:
        conf, iou = 0.10, 0.40
    model_name = input("Modelo YOLOv8 (def=yolov8m.pt): ").strip() or "yolov8m.pt"

    inPath  = Path(INPUT_DIR).resolve()
    cowPath = Path(COW_DIR).resolve()
    noCowPath = Path(NOCOW_DIR).resolve() if NOCOW_DIR else None

    if not inPath.is_dir():
        print(f"❌ La carpeta de entrada no existe: {inPath}")
        return

    cowPath.mkdir(parents=True, exist_ok=True)
    if noCowPath:
        noCowPath.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    files = [p for p in inPath.iterdir() if p.suffix in VALID_EXTS and p.is_file()]
    if not files:
        print("⚠️ No se encontraron imágenes soportadas.")
        return

    cntCow = cntNoCow = 0

    for src in tqdm(files, desc="Detectando vacas"):
        results = model(str(src), device=device, conf=conf, iou=iou, verbose=False)

        has_cow = any(
            model.names[int(box.cls[0].item())] == "cow"
            for r in results
            for box in r.boxes
        )

        if has_cow:
            ok = safeMove(src, cowPath)
            cntCow += int(ok)
        else:
            if noCowPath:
                ok = safeMove(src, noCowPath)
                cntNoCow += int(ok)

    print("\n--- Resumen (vacas) ---")
    print(f"Con vaca  : {cntCow}")
    print(f"Sin vaca  : {cntNoCow}")
    print(f"Procesadas: {len(files)}")


# ============ 3) Aplicar brillo (Pillow) ============

def aplyBrightness():
    """
    Aumenta el brillo de todas las imágenes de una carpeta y guarda en otra.
    """

    print("\n=== 3) Aplicar brillo ===")
    INPUT_DIR = input("📂 Carpeta de entrada: ").strip()
    OUTPUT_DIR = input("📂 Carpeta de salida: ").strip()

    try:
        factor = float(input("Factor de brillo (2 leve, 4 medio, 7 alto) (def=5): ") or "5")
    except ValueError:
        factor = 5

    inPath  = Path(INPUT_DIR).resolve()
    outPath = Path(OUTPUT_DIR).resolve()
    if not inPath.is_dir():
        print(f"❌ La carpeta de entrada no existe: {inPath}")
        return
    outPath.mkdir(parents=True, exist_ok=True)

    files = [p for p in inPath.iterdir() if p.suffix in VALID_EXTS and p.is_file()]
    if not files:
        print("⚠️ No se encontraron imágenes soportadas.")
        return

    ok = 0
    for src in tqdm(files, desc="Aplicando brillo"):
        try:
            img = Image.open(str(src))
            enhancer = ImageEnhance.Brightness(img)
            bright = enhancer.enhance(factor)
            bright.save(str(outPath / src.name))
            ok += 1
        except Exception as e:
            print(f"  ⚠️ Error con {src.name}: {e}")

    print(f"\n✅ Guardadas {ok} imágenes en: {outPath}")


# ============ MAIN (menú) ============

def main():
    while True:
        print("\n============== MENÚ ==============")
        print("1) Separar oscuras / no oscuras")
        print("2) Separar con vaca / sin vaca (YOLOv8 COCO)")
        print("3) Aplicar brillo")
        print("==================================")
        opt = input("Elige una opción (1/2/3): ").strip()

        if opt == "1":
            separarateDark()
            break
        elif opt == "2":
            separarateCows()
            break
        elif opt == "3":
            aplyBrightness()
            break
        else:
            print("❌ Opción inválida. Intenta de nuevo (solo 1, 2 o 3).")

if __name__ == "__main__":
    main()
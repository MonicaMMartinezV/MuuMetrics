import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil
# Correr desde la terminal | python quitar_oscuras.py

# Pedir ubicaciones al usuario-
INPUT_DIR = input("Ingresa la ruta de la carpeta con las imágenes originales: ").strip()
OUTPUT_DIR = input("Ingresa la ruta de la carpeta donde guardar las imágenes limpias: ").strip()
DARK_DIR   = input("Ingresa la ruta de la carpeta donde guardar las imágenes oscuras: ").strip()

# Parámetros | Ajustar al gusto.
BRIGHTNESS_THRESHOLD = 40   # promedio de pixel (0-255), debajo de esto es oscuro | 20 para quitar super oscuras, 40 no tan oscuras
DARK_PERCENT = 0.80         # si el 80% de los pixeles son oscuros -> descartar | 0.95 para quitar super oscuras, 0.80 no tan oscuras

def is_dark_image(img, brightness_threshold=BRIGHTNESS_THRESHOLD, dark_percent=DARK_PERCENT):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # brillo promedio
    avg_brightness = np.mean(gray)
    # porcentaje de pixeles oscuros
    dark_pixels = np.sum(gray < brightness_threshold) / gray.size
    return avg_brightness < brightness_threshold and dark_pixels > dark_percent

# Recorremos todas las imágenes
for fname in tqdm(os.listdir(INPUT_DIR)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path)
        if img is None:
            continue
        if is_dark_image(img):
            shutil.move(path, os.path.join(DARK_DIR, fname))  # mover a descartadas
        else:
            shutil.move(path, os.path.join(OUTPUT_DIR, fname))  # mover a limpias

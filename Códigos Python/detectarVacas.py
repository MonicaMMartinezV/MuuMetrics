import os
import shutil
from ultralytics import YOLO
# Correr desde la terminal | python detectar_vacas_yolo.py

# Pedir rutas al usuario
INPUT_DIR = input("Ingresa la ruta de la carpeta con las imágenes a clasificar: ").strip()
OUTPUT_DIR = input("Ingresa la carpeta destino para imágenes CON vacas: ").strip()
NO_COW_DIR = input("Ingresa la carpeta destino para imágenes SIN vacas (opcional, enter para omitir): ").strip()

# Crear carpetas si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
if NO_COW_DIR:
    os.makedirs(NO_COW_DIR, exist_ok=True)

# Cargar modelo YOLOv8 preentrenado (COCO incluye "cow")
model = YOLO("yolov8m.pt")

# Número de imágenes a procesar (0 = todas)
numImages = 0  # Cambia este valor si quieres limitar

# Listar imágenes válidas
imageFiles = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Si numImages > 0, limitar la lista
if numImages > 0:
    imageFiles = imageFiles[:numImages]

# Contadores
countCow = 0
countNoCow = 0

# Recorremos imágenes
for filename in imageFiles:
    imagePath = os.path.join(INPUT_DIR, filename)

    # Ejecutar la detección
    results = model(imagePath, device="cpu", conf=0.10, iou=0.4)

    # Revisar si alguna detección corresponde a "cow"
    hasCow = any(
        model.names[int(box.cls[0].item())] == "cow"
        for result in results
        for box in result.boxes
    )

    # Mover según detección
    if hasCow:
        shutil.move(imagePath, os.path.join(OUTPUT_DIR, filename))
        countCow += 1
    else:
        if NO_COW_DIR:
            shutil.move(imagePath, os.path.join(NO_COW_DIR, filename))
        countNoCow += 1

# Resumen final
print(f"Imágenes procesadas: {len(imageFiles)}")
print(f"Con vacas: {countCow}")
print(f"Sin vacas: {countNoCow}")
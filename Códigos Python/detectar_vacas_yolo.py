import os
import shutil
from ultralytics import YOLO
# Correr desde la terminal | python detectar_vacas_yolo.py

# Pedir rutas al usuario
input_folder = input("Ingresa la ruta de la carpeta con las imágenes a clasificar: ").strip()
output_folder = input("Ingresa la carpeta destino para imágenes CON vacas: ").strip()
no_cow_folder = input("Ingresa la carpeta destino para imágenes SIN vacas (opcional, enter para omitir): ").strip()

# Crear carpetas si no existen
os.makedirs(output_folder, exist_ok=True)
if no_cow_folder:
    os.makedirs(no_cow_folder, exist_ok=True)

# Cargar modelo YOLOv8 preentrenado (COCO incluye "cow")
model = YOLO("yolov8m.pt")

# Número de imágenes a procesar (0 = todas)
num_images = 0  # Cambia este valor si quieres limitar

# Listar imágenes válidas
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

# Si num_images > 0, limitar la lista
if num_images > 0:
    image_files = image_files[:num_images]

# Contadores
count_cow = 0
count_no_cow = 0

# Recorremos imágenes
for filename in image_files:
    image_path = os.path.join(input_folder, filename)

    # Ejecutar la detección
    results = model(image_path, device="cpu", conf=0.10, iou=0.4)

    # Revisar si alguna detección corresponde a "cow"
    has_cow = any(
        model.names[int(box.cls[0].item())] == "cow"
        for result in results
        for box in result.boxes
    )

    # Mover según detección
    if has_cow:
        shutil.move(image_path, os.path.join(output_folder, filename))
        count_cow += 1
    else:
        if no_cow_folder:
            shutil.move(image_path, os.path.join(no_cow_folder, filename))
        count_no_cow += 1

# Resumen final
print(f"Imágenes procesadas: {len(image_files)}")
print(f"Con vacas: {count_cow}")
print(f"Sin vacas: {count_no_cow}")

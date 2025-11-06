# =============================================================
# Nombre del archivo: CropCow.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 19-10-2025
# Descripción: Funciones para realizar corte a la imagen en 
# base a modelo de YOLO.
# Dependencias: 
#   ultralytics
#   cv2
#   os
#   CowScanner.pt(modelo de corte de YOLO)
# =============================================================

import cv2
import os
from ultralytics import YOLO

# Variables Globales
baseDir    = os.path.dirname(os.path.abspath(__file__))
modelPath  = os.path.join(baseDir, "CowScanner.pt")
model       = YOLO(modelPath)

def cropCow(imagePath, savePath = None):
    """
    Corta la imagen de entrada para solo ver la vaca.

    Args:
        imagePath (str): Dirección de la imagen a cortar.
        save      (str): Dirección de directorio para guardar resultados.

    Returns:
        cropped (cv imag): Imagen cortada.
    """

    # Leer imagen con openCV
    image = cv2.imread(imagePath)

    if image is None:
        print(f"No se pudo cargar la imagen, verifica la ruta: \n {imagePath}")
        return None

    #Tomamos el primer resultado de la prediccion.
    results = model(image)[0]   

    # Verificamos que se genero una bounding box. 
    if len(results.boxes) == 0:
        print(f"No se detectaron botes en la imagen: {os.path.basename(imagePath)}")
        return None

    # Elegir el bounding box con mayor confianza.
    boxes  = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # confianza

    # Recuperar ubicación más confiable y los puntos de su bounding box.
    bestIdx       = scores.argmax()
    x1, y1, x2, y2 = boxes[bestIdx].astype(int)

    # Recortar imagen.
    cropped = image[y1:y2, x1:x2]

    # Si se entrego un directorio se guarda la imagen en una carpeta en esta misma dirección.
    if savePath:
        savePath = os.path.join(savePath,f"{os.path.basename(imagePath)}_Cropped.jpg")
        print(savePath)
        cv2.imwrite(savePath, cropped)

    return cropped

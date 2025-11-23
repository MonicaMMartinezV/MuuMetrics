# =============================================================
# Nombre del archivo: genYoloModel.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 2025-10-20
# Descripción: Generar modelo de YOLO.
# Dependencias: 
#   ultralytics
#   os
# =============================================================

from ultralytics import YOLO
import os

def main():
    """
    Realizar entrenamiento de modelo de YOLO.

    Args:
        None

    Returns:
        None

    """

    model = YOLO('yolov8n.pt')
    model.train(
        data= os.path.join(os.path.dirname(os.path.abspath(__file__)), "DataSet",
                           "data.yaml"),
        epochs=100,
        batch=16,
        imgsz=640,
        device='0',  # o 'cpu' si no encuentra GPU
        name='VacaFModel',
        workers=0 #Se agrego por temas de memoria grafica
    )

if __name__ == '__main__':
    main()
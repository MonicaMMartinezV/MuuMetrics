# =============================================================
# Nombre del archivo: pipline.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: function that generets de df with final results
# Dependencias: Funciones de Calcuulo de DEL, Predicciones del 
#               modelo y rango seguro
# =============================================================

import pandas as pd
from calculoDELvacas.calculoDELvacas import completeDirectory
from model           import dfPredict
from range           import Semaforo

def predictDf(pathCows,pathPatadas,pathImages,checkpoint):
    """
    Generar Df que contiene prediccion de BCS, DEL y semaforos finales

    Args:
        pathCows (str)    : direccion de directorio con la informacion de todas las vacas
        pathPatadas (str) : direccion de patadas.csv
        pathImages (str)  : direccion de directorio con todas las imagenes
        checkpoint (str)  : direccion de .pth del modelo

    Returns:	
        df (DataFrame)    : df que contiene prediccion de BCS, DEL y semaforos finales 
    """
    DEL = completeDirectory(pathCows,pathPatadas,pathImages)
    BCS = dfPredict(pathImages,checkpoint)
    df  = pd.merge(DEL, BCS, on=["img"], how="left")
    df['Semaforo'] = df.apply(lambda fila: Semaforo(fila['BCS'], fila['DEL']), axis=1)
    json_string = df.to_json()
    return json_string

#For tests and see the Json format
"""if __name__ == '__main__':
    pathCows    = r"D:\TEC\IA\B2\Clasificacion proyecto\DATOS VACAS MARZO JUNIO"
    pathPatadas = r"D:\TEC\IA\B2\Clasificacion proyecto\patadas_180725.csv"
    pathImages  = r"D:\TEC\IA\B2\Uncropped and classified (do not eliminate)\2.00"
    checkpoint  = r"D:\TEC\IA\B2\ProyectoFinal\MetricsMuu\final_model.pth"

    lable = predictDf(pathCows,pathPatadas,pathImages,checkpoint)
    print (lable)"""
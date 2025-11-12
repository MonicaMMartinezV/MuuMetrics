# =============================================================
# Nombre del archivo: pipline.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: function that generets de df with final results
# Dependencias: Funcion de discrteisacion
# =============================================================


import pandas as pd
from calculoDELvacas import completeDirectory
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
    return df

if __name__ == '__main__':
    pathCows    = r"D:\TEC\IA\B2\Clasificacion proyecto\DATOS VACAS MARZO JUNIO"
    pathPatadas = r"D:\TEC\IA\B2\Clasificacion proyecto\patadas_180725.csv"
    pathImages  = r"D:\TEC\IA\B2\Clasificacion proyecto\2.00"
    checkpoint  = r"D:\TEC\IA\B2\ProyectoFinal\MetricsMuu\final_model.pth"
    print(predictDf(pathCows,pathPatadas,pathImages,checkpoint))
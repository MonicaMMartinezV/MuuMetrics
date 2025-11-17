# =============================================================
# Nombre del archivo: test.py
# Autor: Ulises Orlando Carrizalez Lerín
# Fecha de creación: 11-11-2025
# Descripción: Realizar una prueba para determinar accuracy
# Dependencias: Calculo de directorio 
# =============================================================

import os
import pandas        as pd
from calculoDELvacas.calculoDELvacas import completeDirectorysInDir
from model           import dfPredictTest
from range           import Semaforo

def predictDfTest(pathCows,pathPatadas,pathImages,checkpoint):
    """
    Generar Df que contiene prediccion de BCS, DEL y semaforos finales de un directorio con
    subdirectorios. (para recolectarlo del formato del dataset)

    Args:
        pathCows (str)    : direccion de directorio con la informacion de todas las vacas
        pathPatadas (str) : direccion de patadas.csv
        pathImages (str)  : direccion de directorio con todas las imagenes
        checkpoint (str)  : direccion de .pth del modelo

    Returns:	
        df (DataFrame)    : df que contiene prediccion de BCS, DEL y semaforos finales 
    """
    DEL = completeDirectorysInDir(pathCows,pathPatadas,pathImages)
    BCS = dfPredictTest(pathImages,checkpoint)
    df  = pd.merge(DEL, BCS, on=["img"], how="left")
    df['Semaforo model'] = df.apply(lambda fila: Semaforo(fila['BCS'], fila['DEL']), axis=1)
    df.rename(columns={'BCS': 'BCS model'}, inplace=True)
    return df

def genTestData(pathImages):
    """
    Recupera los valores etiquetados del BCS por imagen y lo guarda en un df

    Args:
        pathImages (str)  : direccion de directorio con todas las imagenes

    Returns:	
        df (DataFrame)    : df que contiene valores de etiquetas y imagen que le corresponde
    """
    results = []
    for root, _, files in os.walk(pathImages):
        for file in files:
            # Filtrar solo imágenes (puedes ampliar la lista si usas otros formatos)
            if file.lower().endswith(('.jpg')):
                ruta_img = os.path.join(root, file)
                results.append({"img": file, "BCS": float(os.path.basename(os.path.dirname(ruta_img)))})
    df = pd.DataFrame(results)
    return df

def labelDf (pathCows,pathPatadas,pathImages):
    """
    Generar Df que contiene los valores reales de BCS, DEL y semaforos.

    Args:
        pathCows (str)    : direccion de directorio con la informacion de todas las vacas
        pathPatadas (str) : direccion de patadas.csv
        pathImages (str)  : direccion de directorio con todas las imagenes

    Returns:	
        df (DataFrame)    : df que contiene valores reals de BCS, DEL y semaforos finales 
    """
    DEL = completeDirectorysInDir(pathCows,pathPatadas,pathImages)
    BCS = genTestData(pathImages)
    df  = pd.merge(DEL, BCS, on=["img"], how="left")
    df['Semaforo label'] = df.apply(lambda fila: Semaforo(fila['BCS'], fila['DEL']), axis=1)
    df.rename(columns={'BCS': 'BCS Label'}, inplace=True)
    return df

def finalTest(pred,label):
    """
    Compara los resultados de la prediccion contra los datos reales y calcula accuracy
    y genera un .csv con los datos comparados.

    Args:
        pred (DataFrame)    : DF con predicciones y semaforo
        label (DataFrame)   : DF con valores reales

    Returns:	
        None
    """
    dftest   = pd.merge(pred, label[["img","Semaforo label",'BCS Label']], on=["img"], how="left")
    accuracy = (dftest["Semaforo model"] == dftest["Semaforo label"]).mean()
    dftest = dftest[["img","ID","DEL","BCS model","BCS Label","Semaforo model","Semaforo label"]]
    dftest.to_csv(f'test.csv', index=False)
    print(f"Accuracy: {accuracy:.2f}")

if __name__ == '__main__':
    pathCows    = r"D:\TEC\IA\B2\Clasificacion proyecto\DATOS VACAS MARZO JUNIO"
    pathPatadas = r"D:\TEC\IA\B2\Clasificacion proyecto\patadas_180725.csv"
    pathImages  = r"D:\TEC\IA\B2\Uncropped and classified (do not eliminate)"
    checkpoint  = r"D:\TEC\IA\B2\ProyectoFinal\MetricsMuu\final_model.pth"
    
    lable = labelDf(pathCows,pathPatadas,pathImages)
    pred  = predictDfTest(pathCows,pathPatadas,pathImages,checkpoint)
    finalTest(pred,lable)
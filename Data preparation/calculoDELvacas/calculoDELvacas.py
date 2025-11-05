# =============================================================
# Nombre del archivo: Calidad de Datos 34Vacas.py
# Autor: Bárbara Paola Alcántara Vega
# Fecha de creación: 20-10-2025
# Descripción: cálculo de los días en leche más próximos al momento de
#              tomar la imagen de una vaca
# Dependencias: numpy, os, pandas, matplotlib, seaborn, glob, datetime
# =============================================================

import os
import pandas as pd
import glob
import datetime

def combinedDfVacas(path):
    """
    Recuperar Data Frame con informacion de todas las vacas

    Args:
        path (str): direccion de directorio con todos los datos de las vacas
        
    Returns:	
        combinedDf (DataFrame): Data Frame de todas las vacas en formato
    """

    csvFiles = glob.glob(os.path.join(path, "*.csv"))

    dfs = []

    for file in csvFiles:
        try:
            cowId = os.path.splitext(os.path.basename(file))[0]

            # Lee el csv
            dfRaw = pd.read_csv(file, header=None)

            # Encuentra la fila que contiene "Hora de inicio"
            headerRowIndex = dfRaw[dfRaw.apply(lambda r: r.astype(str).str.contains
            ("Hora de inicio").any(), axis=1)].index[0]

            # Usa esa fila como encabezado
            df         = dfRaw[headerRowIndex + 1:].copy()
            df.columns = dfRaw.iloc[headerRowIndex].tolist()


            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Inserta la nueva columna vacaId para colocar los nombres de cada
            df.insert(0, "vacaId", cowId)

            dfs.append(df)

        except IndexError:
            print(f"No se encontró encabezado en {file}")
        except pd.errors.ParserError as e:
            print(f"Error leyendo archivo {file}: {e}")

    # Convierten las fechas y horas de los 34 archivos csv combinados
    # a formato datetime

    combinedDf = pd.concat(dfs, ignore_index=True)
    csv34cows = ["vacaId", "Hora de inicio","Duración (mm:ss)"]
    combinedDf = combinedDf[csv34cows]
    combinedDf.to_csv("combinedVacas.csv", index=False)


    # Cambiar a formato datetime
    combinedDf["Hora de inicio"] = (
        combinedDf["Hora de inicio"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)  # quita espacios dobles por si hay
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
        .str.replace("am.", "AM", regex=False)
        .str.replace("pm.", "PM", regex=False)
        .str.replace(".", "", regex=False)  # elimina puntos sueltos por si hay
    )


    combinedDf["Hora de inicio"] = pd.to_datetime(
        combinedDf["Hora de inicio"],
        dayfirst=True,
        errors="coerce"
    )

    #print(combinedDf.columns)
    combinedDf["duracion_timedelta"] = combinedDf["Duración (mm:ss)"].apply(
    lambda x: datetime.timedelta(minutes=int(x.split(":")[0]), seconds=int(x.split(":")[1]))
    )
    combinedDf["Hora de fin"] = combinedDf["Hora de inicio"] + combinedDf["duracion_timedelta"]
    #Revisar si hay valores nulos después de la limpieza
    #print(combinedDf.isna().sum())

    return combinedDf

def getDateImag(imgName):
    """
    Recuper Date Time de imagen

    Args:
        imgName (str): direccion de imagen
        
    Returns:	
        HoraImagen (DateTime): Date Time de imagen
    """
    imgName = os.path.basename(imgName) #Tomar nombre de la imagen
    datetime_str = imgName.split("_")[0] #Descartar todo despues del _
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%d-%H-%M-%S") #generar DateTime

def getIdImag (dfCows,imgDT):
    """
    Recuperar ID de la imagen

    Args:
        dfCows (DataFrame)   : df con infromacion de ID de vacas y sus ordenños
        imgDT (DateTime)     : DateTime de la imagen
        
    Returns:	
        ID (int): ID de la imagen
    """
    # Ordenar por tiempo (por si acaso)
    dfCows = dfCows.sort_values("Hora de fin").reset_index(drop=True)

    # Filtrar las filas con tiempo menor que s_dt
    antes = dfCows[dfCows["Hora de fin"] < imgDT]

    if not antes.empty:
        return int(antes.iloc[-1]["vacaId"])
    else:
        return None

def DfPatadas(pathPatadas):
    """
    Recuperar Data Frame de patadas y poner en formato

    Args:
        pathPatadas (str): direccion de archivo patadas.csv
        
    Returns:	
        patadasDf (DataFrame): Data Frame de patadas en formato
    """
    patadasDf = pd.read_csv(pathPatadas)

    columnsToKeep = ["Número del animal", "DEL", "Hora Inicio Ordeño"]

    patadasDf = patadasDf[columnsToKeep]

    # Limpieza del texto conteniendo la fecha y hora
    patadasDf["Hora Inicio Ordeño"] = (
        patadasDf["Hora Inicio Ordeño"]
        .astype(str)
        .str.strip()
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
    )

    # Conversión flexible a datetime
    patadasDf["Hora Inicio Ordeño"] = pd.to_datetime(
        patadasDf["Hora Inicio Ordeño"],
        dayfirst=True,   # porque el formato es dd/mm/yyyy
        errors="coerce"  # evita que truene si hay un valor raro
    )
    return patadasDf

def getDEL(dfPatadas,ID,horaImg):
    """
    Calcular dias en leche

    Args:
        dfPatadas (DataFrame): df con infromacion archivo de patadas
        ID (int)             : ID de la imagen
        horaImg (DateTime)   : Hora y dia de la imagen
        
    Returns:	
        DEL (int): dias en leche
    """
    fila     = dfPatadas[dfPatadas["Número del animal"] == int(ID)].iloc[0] #recuperar fecha base
    baseDEL  = int(fila["DEL"]) #separar DEL registrado y hora
    Hora     = fila["Hora Inicio Ordeño"]
    timeDEL  = horaImg - Hora#imagen - Registrado
    return baseDEL + timeDEL.days #Sumar el registrado más el calculado

def completeDirectory(pathCows,pathPatadas,pathImages):
    """
    Recuperar DEL de todas las imagenes de un directorio

    Args:
        pathCows (DataFrame)    : direccion de directorio con la informacion de todas las vacas
        pathPatadas (str) : direccion de patadas.csv
        pathImages (str)  : direccion de directorio con todas las imagenes
        
    Returns:	
        df (DataFrame): Data Frame con todos los DEL
    """
    df   = pd.DataFrame(columns=["img","ID", "DEL"]) #Generar DF
    DfV  = combinedDfVacas(pathCows)
    Df   = DfPatadas(pathPatadas)
    for archivo in os.listdir(pathImages):
        dirImg          = os.path.join(pathImages, archivo)
        dataImg         = getDateImag(dirImg)
        ID              = getIdImag(DfV,dataImg)
        DEL             = getDEL(Df,ID,dataImg)
        df.loc[len(df)] = [dirImg,ID,DEL] #Agregar datos DF
    return df

if __name__ == '__main__':
    pathCows    = r"datosDemonstracion\vacasCSVs"
    pathPatadas = r"datosDemonstracion\patadasDf\patadas_180725.csv"
    pathImages  = r"datosDemonstracion\Imagenes"
    print(completeDirectory(pathCows,pathPatadas,pathImages))
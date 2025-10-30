# =============================================================
# Nombre del archivo: Calidad de Datos 34Vacas Clasificadas.py
# Autor: Bárbara Paola Alcántara Vega (modificado con ChatGPT)
# Fecha de modificación: 28-10-2025
# Descripción: cálculo de los días en leche más próximos al momento
#              de tomar la imagen de una vaca, incluyendo clasificación
#              y flag de desviación del Body Condition Score (BCS)
# =============================================================

import os
import pandas as pd
import glob
import datetime

def combinedDfVacas(path):
    csvFiles = glob.glob(os.path.join(path, "*.csv"))
    dfs = []

    for file in csvFiles:
        try:
            cowId = os.path.splitext(os.path.basename(file))[0]
            dfRaw = pd.read_csv(file, header=None)

            headerRowIndex = dfRaw[dfRaw.apply(
                lambda r: r.astype(str).str.contains("Hora de inicio").any(),
                axis=1
            )].index[0]

            df = dfRaw[headerRowIndex + 1:].copy()
            df.columns = dfRaw.iloc[headerRowIndex].tolist()
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.insert(0, "vacaId", cowId)
            dfs.append(df)

        except IndexError:
            print(f"No se encontró encabezado en {file}")
        except pd.errors.ParserError as e:
            print(f"Error leyendo archivo {file}: {e}")

    combinedDf = pd.concat(dfs, ignore_index=True)
    csv34cows = ["vacaId", "Hora de inicio", "Duración (mm:ss)"]
    combinedDf = combinedDf[csv34cows]

    combinedDf["Hora de inicio"] = (
        combinedDf["Hora de inicio"]
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
        .str.replace("am.", "AM", regex=False)
        .str.replace("pm.", "PM", regex=False)
        .str.replace(".", "", regex=False)
    )

    combinedDf["Hora de inicio"] = pd.to_datetime(
        combinedDf["Hora de inicio"],
        dayfirst=True,
        errors="coerce"
    )

    combinedDf["duracion_timedelta"] = combinedDf["Duración (mm:ss)"].apply(
        lambda x: datetime.timedelta(
            minutes=int(x.split(":")[0]),
            seconds=int(x.split(":")[1])
        )
    )
    combinedDf["Hora de fin"] = combinedDf["Hora de inicio"] + combinedDf["duracion_timedelta"]
    return combinedDf


def getDateImag(imgName):
    imgName = os.path.basename(imgName)
    datetime_str = imgName.split("_")[0]
    return datetime.datetime.strptime(datetime_str, "%Y-%m-%d-%H-%M-%S")


def getIdImag(dfCows, imgDT):
    dfCows = dfCows.sort_values("Hora de fin").reset_index(drop=True)
    antes = dfCows[dfCows["Hora de fin"] < imgDT]
    if not antes.empty:
        return int(antes.iloc[-1]["vacaId"])
    else:
        return None


def DfPatadas(pathPatadas):
    patadasDf = pd.read_csv(pathPatadas)
    columnsToKeep = ["Número del animal", "DEL", "Hora Inicio Ordeño"]
    patadasDf = patadasDf[columnsToKeep]

    patadasDf["Hora Inicio Ordeño"] = (
        patadasDf["Hora Inicio Ordeño"]
        .astype(str)
        .str.strip()
        .str.replace("a. m.", "AM", regex=False)
        .str.replace("p. m.", "PM", regex=False)
        .str.replace("a.m.", "AM", regex=False)
        .str.replace("p.m.", "PM", regex=False)
    )

    patadasDf["Hora Inicio Ordeño"] = pd.to_datetime(
        patadasDf["Hora Inicio Ordeño"],
        dayfirst=True,
        errors="coerce"
    )
    return patadasDf


def getDEL(dfPatadas, ID, horaImg):
    fila = dfPatadas[dfPatadas["Número del animal"] == int(ID)].iloc[0]
    baseDEL = int(fila["DEL"])
    Hora = fila["Hora Inicio Ordeño"]
    timeDEL = horaImg - Hora
    return baseDEL + timeDEL.days


def completeDirectory(pathCows, pathPatadas, pathImages):
    """
    Recuperar DEL, clasificación y flag de BCS de todas las imágenes
    de un directorio con subcarpetas por clase.
    """
    df = pd.DataFrame(columns=["img", "ID", "DEL", "classification", "BCS", "flag"])
    DfV = combinedDfVacas(pathCows)
    DfP = DfPatadas(pathPatadas)

    for classFolder in os.listdir(pathImages):
        classPath = os.path.join(pathImages, classFolder)
        if not os.path.isdir(classPath):
            continue

        for archivo in os.listdir(classPath):
            dirImg = os.path.join(classPath, archivo)

            try:
                dataImg = getDateImag(dirImg)
                ID = getIdImag(DfV, dataImg)
                DEL = getDEL(DfP, ID, dataImg)

                # Extract real BCS value from the folder name (e.g., "2.5" or "3")
                BCS = float(classFolder)

                # Determine normal BCS range based on DEL
                if DEL <= 50:
                    normal_min, normal_max = 3.75, 4.00
                elif DEL <= 150:
                    normal_min, normal_max = 3.25, 3.50
                elif DEL <= 250:
                    normal_min, normal_max = 2.75, 3.25
                elif DEL <= 350:
                    normal_min, normal_max = 2.75, 3.50
                else:
                    normal_min, normal_max = 3.00, 3.75

                # Apply deviation rule: flag if BCS differs by more than 2 from the normal range
                if BCS < (normal_min - 2) or BCS > (normal_max + 2):
                    flag = f"Deviation >2 from normal BCS (expected {normal_min}-{normal_max})"
                else:
                    flag = ""

                df.loc[len(df)] = [dirImg, ID, DEL, classFolder, BCS, flag]

            except Exception as e:
                print(f"Error con imagen {dirImg}: {e}")


    return df


if __name__ == '__main__':
    pathCows = r"calculoDELvacas\datosDemonstracion\vacasCSVs"
    pathPatadas = r"calculoDELvacas\datosDemonstracion\patadasDf\patadas_180725.csv"
    pathImages = r"calculoDELvacas\datosDemonstracion\Imagenes"
    resultDf = completeDirectory(pathCows, pathPatadas, pathImages)
    print(resultDf)
    resultDf.to_csv("DEL_clasificadas_flagBCS.csv", index=False)


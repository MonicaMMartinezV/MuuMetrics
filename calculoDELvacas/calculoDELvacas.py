# -*- coding: utf-8 -*-

# =============================================================
# Nombre del archivo: Calidad de Datos 34Vacas.py
# Autor: Bárbara Paola Alcántara Vega
# Fecha de creación: 20-10-2025
# Descripción: cálculo de los días en leche más próximos al momento de
#              tomar la imagen de una vaca
# Dependencias: numpy, os, pandas, matplotlib, seaborn, glob, datetime
# =============================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime


pathCows = r"" #Actualiza la ruta -- calculoDELvacas/datosDemonstracion
pathPatadas = r"" # Actualiza la ruta -- calculoDELvacas/datosDemonstracion/patadasDf


def calculoDEL (path):

    csvFiles = glob.glob(os.path.join(path, "*.csv"))

    dfs = []

    for file in csvFiles:
        try:
            cowId = os.path.splitext(os.path.basename(file))[0]

            # Lee el dataframe sin tomar en cuenta los encabezados para después
            # asignar como encabezado el ID obtenido del nombre de cada archivo
            # por vaca
            dfRaw = pd.read_csv(file, header=None)

            # Encuentra la fila que contiene "Hora de inicio"
            headerRowIndex = dfRaw[dfRaw.apply(lambda r: r.astype(str).str.contains
            ("Hora de inicio").any(), axis=1)].index[0]

            # Usa esa fila como encabezado
            df = dfRaw[headerRowIndex + 1:].copy()
            df.columns = dfRaw.iloc[headerRowIndex].tolist()


            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

            # Inserta la nueva columna vacaId para colocar los nombres de cada
            # archivo correspondientes al Id de cada vaca
            df.insert(0, "vacaId", cowId)

            dfs.append(df)

        except IndexError:
            print(f"No se encontró encabezado en {file}")
        except pd.errors.ParserError as e:
            print(f"Error leyendo archivo {file}: {e}")

    # Aquí se convierten las fechas y horas de los 34 archivos csv combinados
    # a formato datetime

    combinedDf = pd.concat(dfs, ignore_index=True)
    csv34cows = ["vacaId", "Hora de inicio"]
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

    #Revisar si hay valores nulos después de la limpieza
    print(combinedDf.isna().sum())

    print("="*60)
    print("El nombre de las imágenes viene en el siguiente formato:")
    print(" 2025-05-31-15-08-37_cam0_cap1 " )
    print("Por lo que la lista rawDates es una variable dummy que")
    print("representa las ímagenes como input")
    print("="*60)
    print("\n")

    # Aquí es donde se agregará la conexión a las imágenes si se decide meter todas
    # las imágenes de una vez y convertir sus nombres a fechas
    rawDates = [
        "2025-06-01-21-47-55_cam4_cap3",
        "2025-05-31-15-08-37_cam0_cap1",
        "2025-06-01-05-10-12_cam0_cap1"
    ]

    df = pd.DataFrame({"rawDate": rawDates})
    df["fechaLimpia"] = df["rawDate"].str.extract(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})")

    df["datetime"] = pd.to_datetime(df["fechaLimpia"], format="%Y-%m-%d-%H-%M-%S",
                                    errors='coerce')

    # Verificación y mensaje al usuario
    invalidas = df[df["datetime"].isna()]
    if not invalidas.empty:
        print(f"¡ADVERTENCIA! {len(invalidas)} imagen(es) sin fecha válida:")
        for idx, row in invalidas.iterrows():
            print(f"  - {row['rawDate']}")

    # Se muestran solo las fechas y horas válidas convertidas para revisar si todos
    # los nombres de las imágenes fueron convertidos correctamente
    validas = df[df["datetime"].notna()]
    if not validas.empty:
        print(f"\n{len(validas)} fecha(s) procesada(s) correctamente:")
        print(validas["datetime"])
    else:
        print("No hay fechas válidas para procesar.")

    # Se convieren las fechas y horas de patadasDf a formato datetime para empezar
    # a formar un formato homogéneo a través de los archivos conteniendo los datos

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


    # Se debe realizar la conexión con la lista de imágenes y sus nombres para
    # obtener la fecha objetivo

    # Variable dummy en caso de que se metan imágenes una por una
    targetDatetime = datetime.datetime(2025, 6, 1, 21, 47, 55)
    #2025-06-01-21-47-55_cam4_cap3

    # Calcula la diferencia absoluta en segundos
    combinedDf["diferencia"] = abs(combinedDf["Hora de inicio"] - targetDatetime)

    # Encuentra la fila con la menor diferencia en el dataset de las 34 vacas unidas
    closestRow = combinedDf.loc[combinedDf["diferencia"].idxmin()]

    # Extrae la vaca_id del dataset de 34 vacas unidas y la fecha más cercana a la
    # de la imagen para después utilizarla en el cálculo de DEL a la fecha de la
    # imagen
    vacaId = closestRow["vacaId"]
    foundDate = closestRow["Hora de inicio"]

    print(f"Vaca encontrada: {vacaId}")
    print(f"Fecha más cercana: {foundDate}")

    # Recordatorio de que todas las columnas necesarias deben estar en el df
    # que se le pide adjuntar al usuario
    if 'Número del animal' not in patadasDf.columns or 'DEL' not in patadasDf.columns or 'Hora Inicio Ordeño' not in patadasDf.columns:
        raise ValueError("El DataFrame no tiene las columnas 'Número del animal','DEL' o 'Hora Inicio Ordeño'")

    # Buscar cada vaca convirtiendo vacaId a entero
    cowRow = patadasDf[patadasDf['Número del animal'] == int(vacaId)]

    if cowRow.empty:
        print(f"No se encontró el identificador de la vaca {vacaId} en patadasDf.")
    else:
        for _, row in cowRow.iterrows():
            # Convertir y formatear la hora de ordeño para evitar problmas de tipo
            milkingDate = pd.to_datetime(row['Hora Inicio Ordeño'], errors='coerce')

            # Formatear la fecha y hora para mejor legibilidad
            if pd.notna(milkingDate):
                formattedDate = milkingDate.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formattedDate = "Fecha inválida"

            foundDEL = row['DEL']

            print(f"Vaca: {row['Número del animal']}")
            print(f"Días en Leche (DEL): {row['DEL']} días")
            print(f"Hora de inicio del ordeño: {formattedDate}")
            print("-" * 50)

    # formattedDate (de patadasDf) y foundDate (de combinedDf) se suman o se restan
    # para obtener los días en leche de la vaca a la fecha y hora de la fecha más
    # cercana a la que se tomó su imagen.

    # La razón por la cual no se usa la fecha de la imagen directamente es para que
    # el usuario pueda buscar los datos registrados en relación a la fecha de los
    # datasets de otros aspectos como pezones que presentaron patadas en esa vaca en
    # caso de incorporarlos a futuro.

    if not cowRow.empty and not closestRow.empty:
        date1 = pd.to_datetime(cowRow['Hora Inicio Ordeño'].iloc[0]) #patadas
        date2 = pd.to_datetime(closestRow["Hora de inicio"])         #34 vacas

        # Se obtiene la diferencia de días (positivo o negativo) para restar o sumar
        # a los DEL registrados con la fecha más cercana de a la de la imagen)
        dayDifference = (date1 - date2).days
        print(f"Diferencia entre {date1} y {date2}: {dayDifference} días")

        foundDEL = cowRow['DEL'].iloc[0]

        photoDEL = foundDEL + dayDifference
        print(f"Días en leche más cercanos al momento de tomar la fotografía: {photoDEL}")

    else:
        print("No se encontró la vaca en uno de los DataFrames")


if __name__ -- '__main__':
    main()

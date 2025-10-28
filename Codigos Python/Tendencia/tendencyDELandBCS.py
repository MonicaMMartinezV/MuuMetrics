
# =============================================================
# Nombre del archivo: tndencyDELandBCS.py
# Autor: Bárbara Paola Alcántara Vega
# Fecha de creación: 22-10-2025
# Descripción: análisis de correlación de características de
#              conformación lineal con respecto a la condición
#              corporal (BCS)
# Dependencias: os, pandas, numpy, matplotlib, seaborn, glob,
#               datetime
# =============================================================

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import datetime

pathCows = r"Codigos Python/Tendencia/cows24" #Actualizar path 
pathPatadas = r"Codigos Python/Tendencia/patadas_180725.csv" #Actualizar path
pathBCS = r"C:\Users\baral\OneDrive\Escritorio\Investigación Vacas\bcs\ClassifiedCows" #Actualizar path

#Adaptación del código calculoDELvacas.py para obtener BCS de los archivos de etiquetado
def getAllCowsDELBCS(pathCows, pathPatadas, pathBCS):
    csvFiles = glob.glob(os.path.join(pathCows, "*.csv"))
    dfs = []
    
    print(f"Found {len(csvFiles)} CSV files in {pathCows}")
    
    for file in csvFiles:
        try:
            cowId = os.path.splitext(os.path.basename(file))[0]
            
            # leer df sin titulares
            dfRaw = pd.read_csv(file, header=None)
            
            # encontrar la fila que tiene la "Hora de inicio"
            headerRowIndex = dfRaw[dfRaw.apply(lambda r: r.astype(str).str.contains("Hora de inicio").any(), axis=1)].index[0]
            
            # Usar fila como titular
            df = dfRaw[headerRowIndex + 1:].copy()
            df.columns = dfRaw.iloc[headerRowIndex].tolist()
            
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            
            # POner la nueva columna de vacaID
            df.insert(0, "vacaId", cowId)
            dfs.append(df)
            
        except IndexError:
            print(f"No se encontró encabezado en {file}")
        except pd.errors.ParserError as e:
            print(f"Error leyendo archivo {file}: {e}")
        except Exception as e:
            print(f"Error procesando archivo {file}: {e}")
    
    # combinar todos los csv de cada vaca en uno solo
    if not dfs:
        print("No se pudieron procesar los archivos CSV")
        return pd.DataFrame()
    
    combinedDf = pd.concat(dfs, ignore_index=True)
    
    # solo mantener las columnas necesarias...
    csv34cows = ["vacaId", "Hora de inicio"]
    combinedDf = combinedDf[csv34cows]
    
    # formato datetime ya transformado
    combinedDf["Hora de inicio"] = (
        combinedDf["Hora de inicio"].astype(str).str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"a\.? m\.?", "AM", regex=True)
        .str.replace(r"p\.? m\.?", "PM", regex=True)
        .str.replace(".", "", regex=False)
    )
    combinedDf["Hora de inicio"] = pd.to_datetime(combinedDf["Hora de inicio"], dayfirst=True, errors="coerce")
    
    print(f"Loaded {len(combinedDf)} cow records from {len(csvFiles)} files")
    print(f"Unique cows in combined file: {combinedDf['vacaId'].nunique()}")
    print(f"Sample cow IDs: {combinedDf['vacaId'].unique()[:10]}")
    
    # sacar el puntaje BCS del nombre de los archivos
    bcsData = []
    for bcsFolder in glob.glob(os.path.join(pathBCS, "*")):
        if os.path.isdir(bcsFolder):
            bcsValue = os.path.basename(bcsFolder)
            try:
                bcsValue = float(bcsValue)
            except ValueError:
                continue
                
            imageFiles = glob.glob(os.path.join(bcsFolder, "*.jpg")) + glob.glob(os.path.join(bcsFolder, "*.png"))
            print(f"Found {len(imageFiles)} images in BCS folder: {bcsValue}")
            
            for imgFile in imageFiles:
                imgName = os.path.basename(imgFile)
                # Extract datetime del nombre de la imagen (format: 2025-05-31-15-08-37_cam0_cap1)
                datetimeStr = imgName.split('_')[0]
                try:
                    imgDatetime = pd.to_datetime(datetimeStr, format="%Y-%m-%d-%H-%M-%S")
                    bcsData.append({
                        'datetime': imgDatetime,
                        'BCS': bcsValue,
                        'imageName': imgName
                    })
                except ValueError:
                    continue
    
    if not bcsData:
        print("No BCS data found!")
        return pd.DataFrame()
        
    bcsDf = pd.DataFrame(bcsData)
    print(f"Processed {len(bcsDf)} images with BCS scores")
    
    # Procesar datos de patadas 
    try:
        patadasDf = pd.read_csv(pathPatadas)
        columnsToKeep = ["Número del animal", "DEL", "Hora Inicio Ordeño"]
        
        # Revisar si sí existen las columnas que ando buscando
        missingCols = [col for col in columnsToKeep if col not in patadasDf.columns]
        if missingCols:
            print(f"Missing columns in patadas data: {missingCols}")
            print(f"Available columns: {list(patadasDf.columns)}")
            return pd.DataFrame()
            
        patadasDf = patadasDf[columnsToKeep]
        
        patadasDf["Hora Inicio Ordeño"] = (
            patadasDf["Hora Inicio Ordeño"].astype(str).str.strip()
            .str.replace(r"a\.? m\.?", "AM", regex=True)
            .str.replace(r"p\.? m\.?", "PM", regex=True)
        )
        patadasDf["Hora Inicio Ordeño"] = pd.to_datetime(patadasDf["Hora Inicio Ordeño"], dayfirst=True, errors="coerce")
        
        print(f"Loaded {len(patadasDf)} patadas records")
        print(f"Unique cows in patadas data: {patadasDf['Número del animal'].nunique()}")
        print(f"Sample cow IDs from patadas: {patadasDf['Número del animal'].unique()[:10]}")
        
    except Exception as e:
        print(f"Error loading patadas data: {e}")
        return pd.DataFrame()
    
    print(f"\n=== DEBUG MATCHING ===")
    allCowIdsCombined = set(combinedDf['vacaId'].astype(str))
    allCowIdsPatadas = set(patadasDf['Número del animal'].astype(str))
    
    print(f"Cows in combined data: {len(allCowIdsCombined)}")
    print(f"Cows in patadas data: {len(allCowIdsPatadas)}")
    print(f"Intersection (cows in both): {len(allCowIdsCombined.intersection(allCowIdsPatadas))}")
    
    # Coincidir las vacas con su DEL y BCS
    results = []
    matchStats = {'found': 0, 'notFoundPatadas': 0, 'datetimeMismatch': 0}
    
    for idx, bcsRow in bcsDf.iterrows():
        targetDatetime = bcsRow['datetime']
        bcsValue = bcsRow['BCS']
        
        # Encontrar vaca más cercana
        combinedDf["diferencia"] = abs(combinedDf["Hora de inicio"] - targetDatetime)
        closestRow = combinedDf.loc[combinedDf["diferencia"].idxmin()]
        cowId = closestRow["vacaId"]
        foundDate = closestRow["Hora de inicio"]
        
        # Encontrar DEL para una vaca específica
        try:
            # Convertir cow_id para coincidir con el formato de patadas
            cowRow = patadasDf[patadasDf['Número del animal'].astype(str) == str(cowId)]
            
            if not cowRow.empty:
                date1 = pd.to_datetime(cowRow['Hora Inicio Ordeño'].iloc[0])
                date2 = pd.to_datetime(foundDate)
                
                # Solo incluir si las fechas tienen sentido (1 año)
                if abs((date1 - date2).days) < 365:
                    dayDifference = (date1 - date2).days
                    foundDel = cowRow['DEL'].iloc[0]
                    photoDel = foundDel + dayDifference
                    
                    results.append({
                        'cowId': cowId,
                        'DEL': photoDel,
                        'BCS': bcsValue,
                        'imageName': bcsRow['imageName'],
                        'matchedDate': foundDate
                    })
                    matchStats['found'] += 1
                else:
                    matchStats['datetimeMismatch'] += 1
            else:
                matchStats['notFoundPatadas'] += 1
                
        except Exception as e:
            print(f"Error processing cow {cowId}: {e}")
            continue
    
    print(f"\n=== MATCHING RESULTS ===")
    print(f"Successful matches: {matchStats['found']}")
    print(f"Cows not found in patadas data: {matchStats['notFoundPatadas']}")
    print(f"Date mismatches (>1 year): {matchStats['datetimeMismatch']}")
    
    if not results:
        print("No matches found between BCS images and cow data!")
        return pd.DataFrame()
    
    # Crear el df final y quitar duplicados que se pudieran haber escapado 
    finalDf = pd.DataFrame(results)
    print(f"\nBefore removing duplicates: {len(finalDf)} records")
    finalDf = finalDf.drop_duplicates().reset_index(drop=True)
    print(f"After removing duplicates: {len(finalDf)} records")
    print(f"Unique cows in final results: {finalDf['cowId'].nunique()}")
    print(f"Sample final cow IDs: {finalDf['cowId'].unique()[:10]}\n")
    
    return finalDf

# función para analizar la consistencia de puntuaciones BCS por vaca
def analyzeBCSConsistency(df):
    print("\n" + "="*80)
    print("ANÁLISIS DE CONSISTENCIA DE BCS POR VACA")
    print("="*80)
    
    # Agrupar por vaca y analizar su puntaje de BCS 
    cowBcsStats = df.groupby('cowId')['BCS'].agg([
        'count', 
        'mean', 
        'std', 
        'min', 
        'max',
        lambda x: x.max() - x.min()  # rango
    ]).round(3)
    
    cowBcsStats.columns = ['nImagenes', 'BCSPromedio', 'BCSStd', 'BCSMin', 'BCSMax', 'BCSRango']
    
    # Mostrar el resumen general de imágenes y puntajes por vaca para ver más o menos
    #cuántas imágenes etiquetadas por vaca hay y cuántas solo tienen una instancia de puntaje
    print(f"Resumen general:")
    print(f"Total de vacas únicas: {len(cowBcsStats)}")
    print(f"Total de imágenes: {df.shape[0]}")
    print(f"Promedio de imágenes por vaca: {df.shape[0] / len(cowBcsStats):.1f}")
    
    # Mostrar distribución de vacas por número de imágenes para ver más o menos
    #cómo se comportaron nuestros puntajes por vaca
    imageCountDist = cowBcsStats['nImagenes'].value_counts().sort_index()
    print(f"\nDistribución de vacas por número de imágenes:")
    for count, freq in imageCountDist.items():
        print(f"  {count} imagen(es): {freq} vaca(s)")
    
    return cowBcsStats


#función que grafica el DEL en el eje x y BCS en el eje 
def plotDELvsBCSComplete(df):

    plt.figure(figsize=(14, 8))
    
    # Crear scatter plot con diferentes colores para rangos de BCS
    colors = plt.cm.viridis((df['BCS'] - df['BCS'].min()) / (df['BCS'].max() - df['BCS'].min()))
    scatter = plt.scatter(df['DEL'], df['BCS'], c=colors, alpha=0.7, s=60, edgecolor='black', linewidth=0.5)
    
    # Agregar línea de tendencia (quadratica para capturar la potential U-shape)
    z = np.polyfit(df['DEL'], df['BCS'], 2)
    p = np.poly1d(z)
    xTrend = np.linspace(df['DEL'].min(), df['DEL'].max(), 100)
    plt.plot(xTrend, p(xTrend), "r-", linewidth=3, label='Tendencia cuadrática')
    
    # Agregar etiquetas y título
    plt.xlabel('Días en Leche (DEL)', fontsize=14, fontweight='bold')
    plt.ylabel('Body Condition Score (BCS)', fontsize=14, fontweight='bold')
    plt.title('Relación entre Días en Leche y Body Condition Score\n30 Vacas - 644 Imágenes', 
              fontsize=16, fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    plt.ylim(1.5, 5.5)
    
    # agregar colorbar para el BCS
    cbar = plt.colorbar(scatter)
    cbar.set_label('BCS', fontsize=12)
    
    # agregar estadísticos 
    correlation = df['DEL'].corr(df['BCS'])
    plt.text(0.02, 0.98, f'Correlación: {correlation:.3f}\nN = {len(df)} puntos\n{df["cowId"].nunique()} vacas', 
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
             verticalalignment='top')
    
    # Agregar los nombres de cada etapa de lactación
    plt.axvspan(0, 100, alpha=0.1, color='red', label='Lactancia temprana\n(BCS ↓)')
    plt.axvspan(100, 200, alpha=0.1, color='orange', label='Lactancia media')
    plt.axvspan(200, 500, alpha=0.1, color='green', label='Lactancia tardía\n(BCS ↑)')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    return correlation


# Función para el análisis de tendencia de BCS por etapas de lactancia (con curva)
def analyzeTrendByLactationStage(df):
    
    # Definir etapas de lactación (3, temprana, media, tardía)
    early = df[df['DEL'] <= 100]
    mid = df[(df['DEL'] > 100) & (df['DEL'] <= 200)]
    late = df[df['DEL'] > 200]
    
    stages = [
        ("Lactancia temprana (0-100 DEL)", early),
        ("Lactancia media (101-200 DEL)", mid), 
        ("Lactancia tardía (>200 DEL)", late)
    ]
    
    for stageName, stageData in stages:
        if len(stageData) > 0:
            print(f"{stageName}:")
            print(f"  {len(stageData)} puntos, {stageData['cowId'].nunique()} vacas")
            print(f"  BCS promedio: {stageData['BCS'].mean():.3f} ± {stageData['BCS'].std():.3f}")
            print(f"  Rango DEL: {stageData['DEL'].min()}-{stageData['DEL'].max()}")
        else:
            print(f"{stageName}: No hay datos")
        print()
    
    # Se calculan las tendencias sólo si hay datos 
    if len(early) > 0 and len(late) > 0:
        trendEarlyToLate = late['BCS'].mean() - early['BCS'].mean()
        print(f"Cambio en BCS (temprano → tardío): {trendEarlyToLate:+.3f}")
        
        if trendEarlyToLate > 0.1:
            print("TENDENCIA DETECTADA: BCS se recupera en lactancia tardía")
        elif trendEarlyToLate < -0.1:
            print("TENDENCIA: BCS disminuye hacia lactancia tardía")
        else:
            print("TENDENCIA: BCS se mantiene estable")
    else:
        print("No hay datos suficientes para analizar la tendencia temprano-tardío")

def plotIndividualCowTrajectories(df):
    """
    Plot individual cow BCS trajectories over DEL
    """
    # Agarra vacas con varias medidas para un análisis de trayectoria
    # Algunas tienen solo una medida y esas vacas no nos dan una
    #buena idea de cómo calificamos el BCS
    cowCounts = df.groupby('cowId').size()
    cowsWithMultiple = cowCounts[cowCounts > 3].index
    
    if len(cowsWithMultiple) > 0:
        plt.figure(figsize=(12, 8))
        
        # Agregar colormap para distinguir etapas mejor
        colors = plt.cm.tab20(np.linspace(0, 1, min(15, len(cowsWithMultiple))))
        
        for i, cowId in enumerate(cowsWithMultiple[:15]):  # Plotear primeras 15 para ver cómo las etieuqtamos
            cowData = df[df['cowId'] == cowId].sort_values('DEL')
            plt.plot(cowData['DEL'], cowData['BCS'], 'o-', linewidth=2, markersize=6, 
                    color=colors[i], label=f'Vaca {cowId}', alpha=0.8)
        
        plt.xlabel('Días en Leche (DEL)', fontsize=12, fontweight='bold')
        plt.ylabel('Body Condition Score (BCS)', fontsize=12, fontweight='bold')
        plt.title('Trayectorias Individuales de BCS por Vaca\n(Vacas con múltiples mediciones)', 
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(0, max(500, df['DEL'].max() + 10))
        plt.ylim(1.5, 5.5)
        plt.tight_layout()
        plt.show()
        
        print(f"\nSe muestran {min(15, len(cowsWithMultiple))} vacas con múltiples mediciones de BCS")
        
        # Para ver más o menos como varían los puntajes en algunasvacas...
        print(f"\nEstadísticas de vacas con trayectorias:")
        for cowId in cowsWithMultiple[:5]:  # mostrar primeras 5 vacas como ejemplo
            cowData = df[df['cowId'] == cowId].sort_values('DEL')
            delRange = f"{cowData['DEL'].min()}-{cowData['DEL'].max()}"
            bcsRange = f"{cowData['BCS'].min():.1f}-{cowData['BCS'].max():.1f}"
            print(f"Vaca {cowId}: DEL {delRange}, BCS {bcsRange}, {len(cowData)} mediciones")
            
    else:
        print("\nNo hay suficientes vacas con múltiples mediciones para mostrar trayectorias")

#Función para graficar el puntaje de BCS por los rangos de días en leche (DEL) 
def plotBCSDistributionByDEL(df):
    
    # Crear categorías o bins de DEL
    df['DELBin'] = pd.cut(df['DEL'], bins=[0, 50, 100, 150, 200, 300, 500])
    
    plt.figure(figsize=(12, 6))
    
    # Box plot
    plt.subplot(1, 2, 1)
    delBinsOrdered = ['(0, 50]', '(50, 100]', '(100, 150]', '(150, 200]', '(200, 300]', '(300, 500]']
    bcsByDel = []
    labels = []
    
    for binRange in delBinsOrdered:
        binData = df[df['DELBin'].astype(str) == binRange]['BCS']
        if len(binData) > 0:
            bcsByDel.append(binData)
            # covertir las etiquetas de los bins", cubitos, rectangulitos, categorías
            cleanLabel = binRange.replace('(', '').replace(']', '').replace(',', '-')
            labels.append(cleanLabel)
    
    if bcsByDel:  # SOLO se grafica si sí hay datos
        boxPlot = plt.boxplot(bcsByDel, tick_labels=labels)
        plt.xlabel('Rango de DEL', fontweight='bold')
        plt.ylabel('BCS', fontweight='bold')
        plt.title('Distribución de BCS por Rango de DEL', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No hay datos suficientes\npara el box plot', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    # Violin plot para ver la densidad de probabilidad
    plt.subplot(1, 2, 2)
    plotData = []
    for i, label in enumerate(labels):
        for value in bcsByDel[i]:
            plotData.append({'DELRange': label, 'BCS': value})
    
    if plotData:  # SOLO graficar, si sí hay datos (flexible en caso de cambios)
        plotDf = pd.DataFrame(plotData)
        sns.violinplot(data=plotDf, x='DELRange', y='BCS')
        plt.xlabel('Rango de DEL', fontweight='bold')
        plt.ylabel('BCS', fontweight='bold')
        plt.title('Distribución de Densidad de BCS', fontweight='bold')
        plt.xticks(rotation=45)
    else:
        plt.text(0.5, 0.5, 'No hay datos suficientes\npara el violin plot', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Estadísticas por cada... "bin", por cada cajita
    print(f"\n=== ESTADÍSTICAS POR RANGO DE DEL ===")
    for i, label in enumerate(labels):
        if i < len(bcsByDel):
            data = bcsByDel[i]
            if len(data) > 0:
                print(f"{label} DEL: {len(data)} puntos, BCS: {data.mean():.3f} ± {data.std():.3f}")


if __name__ == '__main__':
    resultDf = getAllCowsDELBCS(pathCows, pathPatadas, pathBCS)
    if not resultDf.empty:

        
        print(f"\nTotal de registros: {len(resultDf)}")
        print(f"Total de vacas únicas: {resultDf['cowId'].nunique()}")
        print(f"Rango de DEL: {resultDf['DEL'].min()} - {resultDf['DEL'].max()}")
        print(f"Rango de BCS: {resultDf['BCS'].min()} - {resultDf['BCS'].max()}")
        
        # Guardar los resultados en un nuevo archivo en caso de que sea necesario
        resultDf.to_csv("cows_DEL_BCS.csv", index=False)
        print(f"\nDatos guardados en cows_DEL_BCS.csv")
        
        # Análisis de consistencia UwU
        cowBcsStats = analyzeBCSConsistency(resultDf)
        
        
        correlation = plotDELvsBCSComplete(resultDf)
        analyzeTrendByLactationStage(resultDf)
        plotIndividualCowTrajectories(resultDf)
        plotBCSDistributionByDEL(resultDf)
        
        print(f"\n Correlación final DEL-BCS: {correlation:.3f}")
        
    else:
        print("No hay resultados para analizar.")

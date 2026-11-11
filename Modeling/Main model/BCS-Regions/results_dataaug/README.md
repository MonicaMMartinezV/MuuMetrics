# Resultados del estudio de data augmentation (etapas A, B y C)

Copia publicada de las tablas y figuras finales del estudio descrito en
[`../INFORME_DATA_AUGMENTATION.md`](../INFORME_DATA_AUGMENTATION.md) (informe) y
[`../ABLATION.md`](../ABLATION.md) (metodología). Los archivos se copiaron **sin
modificar** desde `work/aug/` (carpeta local, ignorada por Git); no se recalculó ninguna
métrica. La única excepción es `nested_A0A1/report_aggregate.md` (ver abajo).

## A qué versión del modelo pertenecen

| | |
|---|---|
| Modelo canónico evaluado | `model/main` @ `47b515d` (BCS-Regions con **DINOv2**) |
| Segmentador | DINOv2-B/14 + FPN, 6 regiones (`loin`, `hips`, `rump`, `tailhead`, `thigh_L`, `thigh_R`), pesos `final_all40.pt`, TTA con flip |
| Descriptor | DINOv2-L/14 congelado, 378×378, media de tokens por región + indicador de ausencia = 6150 dims |
| Regresión | Ridge estandarizado; α ∈ {1, …, 1e5} elegido con `GroupKFold(4)` interno por vaca; **sin calibración lineal** |
| Datos | 369 fotos, 37 vacas (`Dataset_Fr_BCS` de esa fecha) |
| Validación | LOCO (`GroupKFold(37)`) y `StratifiedGroupKFold(5)` × 10 semillas (0–9), por vaca |
| Código del estudio | rama `model/dataaug` (`augment.py`, `ablation.py`, `nested.py`, `paper_figure.py`), corridas del 2026-11-09/10 |

**No son resultados de versiones posteriores de `model/main`** (p. ej. la calibración
lineal, el dataset de 542 fotos/64 vacas o el modelo con DINOv3-L/16). No deben
compararse directamente con métricas de esas versiones.

## Contenido

| Carpeta | Archivo | Qué es |
|---|---|---|
| `final/` | `table_compact.{md,csv,tex}` | Tabla de publicación: MAE y ΔMAE vs A0 [IC 95 %] por política, en LOCO y 5-fold × 10, más la selección anidada |
| | `table_full.md` | Tabla completa A + B: MAE por foto y por vaca, RMSE, R², sesgo, % dentro de ±0.5, ΔMAE y vacas mejor/peor |
| | `figure_ablation.{pdf,png}` | Figura final (300 dpi): ΔMAE con IC 95 % por política y esquema |
| `report/` | `comparison.{csv,md}`, `delta_mae_vs_A0.{pdf,png}` | Etapa A: A0, A0-DUP, A1–A6 |
| `report_stageB/` | `comparison.{csv,md}`, `delta_mae_vs_A0.{pdf,png}` | Etapa B: A1 con 11 semillas y políticas fotométricas aisladas (B-BR05/10, B-CT05/10) |
| | `stability_A1.{md,pdf,png}`, `stability_A1_summary.csv` | Estabilidad de A1 entre semillas de augmentation |
| `nested_A0A1/` | `comparison.csv` | Etapa C: A0, "A1 siempre" y selección anidada (δ = 0 primario, δ = 0.01 sensibilidad) |
| | `summary.json` | Conteo de folds en que se eligió A0 o A1 y controles de reproducción |
| | `prereg.json` | Diseño pre-registrado, fecha y hashes del código y de los descriptores |
| | `leakage_check.json` | Verificación de que ninguna vaca de test entra en el train interno |
| | `report_aggregate.md` | Primeras 12 líneas de `report.md` (selecciones, controles y tabla agregada) |

## Cómo leer las tablas

- **ΔMAE = MAE(política) − MAE(A0)**, pareado: mismas vacas, mismos folds y mismo α.
  Negativo = menor error que sin augmentation.
- **IC 95 %**: bootstrap por vaca (5000 remuestreos, semilla 0).
- **MAE (cow)**: primero se promedia el error de cada vaca y luego entre vacas.
- **n.s.**: el IC de ΔMAE incluye 0 en los cuatro análisis (foto/vaca × LOCO/5-fold × 10).
  **†**: el IC excluye 0 en al menos uno; sin ajuste por multiplicidad.
- **Cows better/worse** (`vacas_mejor`/`vacas_peor`): número de vacas cuyo MAE baja o sube frente a A0.
- **A0-DUP** es un control: duplica las filas sin transformar y debe dar Δ = 0.
- La etapa C es **exploratoria**: A1 se eligió como candidato después de ver las etapas A y B
  en las mismas 37 vacas.

Conclusión del estudio (informe, §10): ninguna política mejora de forma significativa; el flip
horizontal (A1) da Δ ≈ −0.007 con IC que incluye 0, y la rotación (A3) empeora.

## Qué no se publica

Por privacidad y tamaño se quedan solo en `work/aug/` local:

- fotografías y ejemplos visuales (`qc/`), descriptores `.npz` (`features/`), pesos y dataset;
- predicciones por foto (`results/`, `nested_A0A1/preds_nested_loco.csv`, `folds.json`) y
  `splits.json`, que contienen nombres de imágenes e identificadores de vacas;
- tablas por animal (`nested_A0A1/per_cow.csv` y la sección por vaca de `report.md`).

## Trazabilidad

- Todos los archivos, salvo `report_aggregate.md`, son idénticos byte a byte a los de
  `work/aug/` (verificado con `cmp`). `report_aggregate.md` es un prefijo exacto de
  `work/aug/nested_A0A1/report.md`.
- `prereg.json` registra el hash de `nested.py` del commit `33e2bcd`, con el que se corrió
  la etapa C. El commit posterior `5327f52` solo cambia cómo se escribe la tabla por vaca,
  que no se publica aquí; no afecta a ningún archivo de esta carpeta.
- Para regenerar: comandos en `../ABLATION.md` (`ablation.py`, `nested.py`) y
  `paper_figure.py` (escribe en `work/aug/final/`).

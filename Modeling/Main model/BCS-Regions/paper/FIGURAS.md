# Figuras, tablas y datos del manuscrito

Todo lo que usa el manuscrito está en esta rama (`model/main`): figuras en `../figures/`,
tablas de datos en `datos/`. Cada figura indica de dónde sale para poder regenerarla.

## Figuras

| Figura | Archivo (`../figures/`) | Sección | Cómo se genera (rama · script) | EXPERIMENTOS |
|---|---|---|---|---|
| Arquitectura del modelo | `model_bcs_v3.{pdf,png}` | Métodos | main · `pdflatex model_bcs_v3.tex` | [32] |
| Anotaciones de regiones (Roboflow) | `segmentation_examples.{pdf,png}` | Métodos | main · `figures/plot_seg_examples.py` | [15] |
| Acuerdo con la referencia (predicho vs real, Bland–Altman) | `analisis_bland_altman.{pdf,png}` | 3.1 | ablations · `analisis/analisis_paper.py` | [31] |
| Curvas ROC (vacas flacas / gordas) | `roc.{pdf,png}` | 3.3 o suplementario | baselines · `baselines/roc.py` | [28] |
| Seguimiento de vacas con cambio de BCS | `analisis_cambios_bcs.{pdf,png}` | 3.4 | ablations · `analisis/analisis_paper.py` | [31] |
| Curva de aprendizaje | `curva_aprendizaje.{pdf,png}` | 3.4 / discusión | ablations · `exp_curva.py` + `analisis/plot_curva.py` | [31] |
| Curvas de entrenamiento por época de los baselines | `curvas_entrenamiento.{pdf,png}` (llega al terminar la cola) | suplementario | baselines · `ajuste.py --curvas`, `yolo_cls.py --curvas`, `baselines/plot_curvas.py` | [28] |

## Datos (`datos/`)

| Archivo | Contenido |
|---|---|
| `comparacion.csv` | Tabla de baselines y experimento de fuga (división por vaca y por foto) |
| `roc.csv` | AUC [IC95], diferencia contra el modelo, sensibilidad y especificidad por método |
| `error_por_bcs.csv` | Error y sesgo por valor de BCS (LOCO) |
| `estabilidad.csv`, `cambios_bcs.csv` | Estabilidad entre días y vacas con cambio de BCS |
| `curva_aprendizaje.csv` | Curva de aprendizaje por semilla, fold y número de vacas |
| `resumen.md` | Resumen de los análisis de [31] (la sección de cámaras no es válida) |
| `curvas/` | Curvas por época de los baselines (llega al terminar la cola) |

Los resultados del modelo canónico v3 (LOCO y 5×10) están en `../work/metrics.csv`
(regenerables con `docker compose run --rm bcs evaluate`).

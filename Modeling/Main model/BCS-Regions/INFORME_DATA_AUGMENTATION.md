# Data augmentation convencional sobre un modelo de BCS con descriptores DINOv2 congelados: informe final

**Proyecto:** MuuMetrics, estimación de la condición corporal (BCS) de vacas Holstein a
partir de fotos dorsales del robot de ordeño.
**Rama:** `model/dataaug`. **Estado:** estudio cerrado (etapas A, B y C).
**Alcance:** este informe usa solo resultados obtenidos y guardados en
`work/aug/` (no versionado). El código y la metodología detallada están en
`augment.py`, `ablation.py`, `nested.py`, `paper_figure.py` y `ABLATION.md`.

---

## Resumen

Se evaluaron diez políticas de data augmentation solo en entrenamiento, en un modelo
que estima el BCS con descriptores regionales de un DINOv2-L congelado y una regresión
ridge. Se usaron 369 fotos de 37 vacas, validación por vaca (LOCO y 5 folds × 10
semillas) y diferencias pareadas con intervalos de confianza (IC) por bootstrap de
vacas.

- **Ninguna política redujo el error de forma estadísticamente respaldada.**
- **El flip horizontal** produjo la única reducción del MAE: −0.007 BCS en LOCO y
  −0.005 en 5×10. Es reproducible en 11/11 semillas de augmentation, pero todos sus
  IC 95 % incluyen 0.
- **La rotación (±5°), las transformaciones combinadas y la fotometría** aumentaron el
  error. De los 40 IC de diferencia evaluados, 8 excluyen 0 y los 8 indican mayor error.
- **La selección anidada A0/A1 por vaca** eligió A1 en 37/37 folds y no mejoró a A0
  de forma demostrable.

Se mantiene el modelo sin augmentation (A0) como modelo principal.

---

## 1. Modelo canónico (DINOv2 + ridge)

Las 6 regiones anatómicas son lomo, cadera, grupa, base de la cola, muslo izquierdo y
muslo derecho. El modelo tiene tres componentes, y solo el tercero se entrena con BCS:

1. **Segmentador de regiones.** Encuentra las 6 regiones, que pueden traslaparse.
   - Arquitectura: DINOv2-B/14 ajustado con un decodificador FPN.
   - Entrenamiento: 40 imágenes anotadas; mIoU de 0.745 en su conjunto de prueba.
   - Entrada: cada recorte se centra en un lienzo 16:9 y se reescala a 378×672 px.
   - TTA: promedia la predicción de la imagen original y de la volteada, intercambiando
     los canales de muslo izquierdo y derecho.
   - Las máscaras se guardan en caché.
2. **Descriptor.** DINOv2-L/14 (`lvd142m`) **congelado**, sin ajuste fino.
   - La foto se reescala a 378×378 px y se normaliza con la media y desviación de
     ImageNet; resultan 27×27 tokens de parche de 1024 dimensiones.
   - Por región, los tokens se promedian ponderados por la fracción de cada parche
     dentro de la máscara (umbral 0.5).
   - Se agrega un indicador de región ausente (menos de medio parche).
   - Resultado: un vector de 6 × 1024 + 6 = **6150** componentes por foto.
3. **Regresión ridge.**
   - Las columnas se estandarizan con la media y la desviación del train.
   - Se resuelve en forma dual (kernel lineal, n ≪ p); el intercepto no se penaliza.
   - α ∈ {1, 10, 10², 10³, 10⁴, 10⁵} se elige dentro del train de cada fold con
     `GroupKFold(4)` por vaca, minimizando el MAE.
   - α elegido en LOCO: 10³ en 31 folds, 10² en 5 y 1 en 1.
   - α elegido en 5×10: 10³ en 35, 10² en 7, 1 en 6 y 10⁴ en 2.

**Rendimiento de referencia (A0):**

| Esquema | MAE | RMSE | R² fotos | R² vacas |
|---|---|---|---|---|
| LOCO | 0.583 | 0.682 | 0.55 | 0.52 |
| 5 folds × 10 | 0.629 | 0.729 | 0.48 | 0.46 |
| Predecir siempre la media del train (LOCO) | 0.954 | | | |

## 2. Políticas A0–A6 y estado del arte

### 2.1 Literatura consultada

| Referencia | Qué se verificó | Augmentation |
|---|---|---|
| Yao et al. (2026). *Automated dairy cattle body condition score using side-view images and deep learning.* J. Dairy Sci. 109(9):9446–9458. doi:10.3168/jds.2025-27759 | Metadatos y resumen (Crossref, Europe PMC) | El resumen no la menciona; texto completo no accesible (suscripción) |
| Siachos et al. (2024). *Development and validation of a fully automated 2-dimensional imaging system generating body condition scores for dairy cows using machine learning.* J. Dairy Sci. 107(4):2499–2511. doi:10.3168/jds.2023-23894 | Metadatos y resumen | El resumen no la menciona; texto completo no accesible |
| Swastika (2025). JAMRIS, doi:10.14313/jamris-2025-038 | No se pudieron obtener ni metadatos ni texto | Desconocida |
| Bär et al. (2024). *Frozen Feature Augmentation for Few-Shot Image Classification.* CVPR 2024, arXiv:2403.10519 | Texto de arXiv leído hasta la Tabla 2 | Sí (ver abajo) |

**Yao et al. (2026)**, según su resumen:
- 3,208 imágenes **laterales** de 211 vacas Holstein;
- detección con YOLOv11 y regresión con ConvNeXt-Tiny;
- MAE de 0.41 y r = 0.62 con validación de 5 folds estratificada por vaca.

**Siachos et al. (2024)**, según su resumen:
- un sistema 2D automático con regresión ordinal y deep learning;
- 34,150 BCS de entrenamiento en 7 granjas y 9,657 de prueba en 4 granjas.

Ninguno de los dos resúmenes describe la augmentation. Por eso **no se atribuye a esos
trabajos ningún parámetro de augmentation**.

**Bär et al. (2024)** aplican augmentation sobre los tokens del último bloque de ViT
congelados, con una cabeza entrenada sobre ellos, que es un esquema análogo al nuestro.
En su Tabla 2 (L/16 JFT-3B, ImageNet, 1 y 5 shots):
- las geométricas (rotación, shear y traslación) empeoran en todos los casos;
- las puntuales estilísticas (brillo, contraste y posterize) mejoran.

Es la única evidencia publicada verificada que se usa para contrastar nuestros
resultados.

### 2.2 Políticas evaluadas

**Todas las magnitudes son adaptaciones de este estudio**: ninguna proviene de un
parámetro publicado verificado. La justificación es el escenario de captura: cámara
cenital fija del robot (VMS 3), vista dorsal casi simétrica, dos cámaras (cam0 y cam4)
e iluminación variable.

| Política | Transformación por variante | Justificación (nuestra) |
|---|---|---|
| A0 | Ninguna | Baseline |
| A1 | Flip horizontal, p = 0.5, con intercambio de máscaras de los muslos | Simetría bilateral de la vista dorsal |
| A2 | Con p = 0.5: brillo × U(0.9, 1.1) y contraste × U(0.9, 1.1) | Diferencias de iluminación y entre cámaras |
| A3 | Rotación U(−5°, 5°) | Pequeñas variaciones de orientación en el box |
| A4 | Traslación U(−3 %, 3 %) y escala U(0.97, 1.03) | Pequeñas variaciones de posición y distancia |
| A5 | A1 + A2 + A3 + A4 | Combinación |
| A6 | A5 + ruido gaussiano σ = 0.02 (escala 0–1) | Ruido de sensor |
| B-BR05 / B-BR10 | Brillo ± 5 % / ± 10 % (p = 0.5) | Aislar el brillo de A2 |
| B-CT05 / B-CT10 | Contraste ± 5 % / ± 10 % (p = 0.5) | Aislar el contraste de A2 |
| A0-DUP | 3 copias idénticas | Control de duplicación y ponderación |

## 3. Diseño experimental

- **Datos.**
  - Se usa solo `Dataset/Dataset_Fr_BCS`, con vaca y BCS confirmados a mano: **369 fotos
    de 37 vacas**.
  - BCS de 2.00 a 5.00 en pasos de 0.25; por vaca, 8 con BCS < 3, 5 entre 3 y 4, y 24 ≥ 4.
  - Fotos por vaca: de 1 a 48 (mediana 5).
  - 36 vacas tienen un único BCS; una (8758) tiene BCS mensual (se usa el del mes de
    cada foto).
- **Augmentation solo en entrenamiento.**
  - Por foto se agregan K = 3 variantes; su descriptor se calcula con el mismo DINOv2-L
    congelado.
  - Las transformaciones geométricas se aplican con la misma matriz a la foto y a sus
    6 máscaras.
  - El test usa siempre los descriptores originales, sin TTA nuevo.
- **Semillas de augmentation.**
  - Cada variante tiene la suya: `SeedSequence([semilla, crc32(foto), k])`.
  - Los parámetros se sortean siempre y la política elige cuáles usa. Las políticas
    quedan **anidadas**: A5 usa los mismos flips que A1, y B-BR10 y B-CT10 son
    exactamente las componentes de A2.
- **Validación**, siempre por vaca y con las mismas particiones para todas las
  políticas:
  - LOCO: 37 folds.
  - 5 folds por vaca × 10 semillas (`StratifiedGroupKFold`, estratificado por BCS).
    En este esquema las predicciones de las 10 repeticiones se promedian por foto.
- **Métricas:**
  - MAE por foto y MAE promedio por vaca;
  - RMSE, R² por foto y por vaca, bias, y porcentaje de fotos con error ≤ 0.5 BCS
    (predicción redondeada a 0.25);
  - ΔMAE pareado contra A0 con IC 95 % por bootstrap de **vacas** (5000 réplicas,
    semilla 0; las mismas vacas remuestreadas para todas las políticas).

## 4. Controles

| Control | Resultado |
|---|---|
| A0 reproduce las predicciones canónicas | Diferencia máxima 5 × 10⁻⁵ (los CSV canónicos tienen 4 decimales) |
| A0-DUP (filas duplicadas) = A0 | Diferencia máxima 2 × 10⁻¹⁴ en todas las repeticiones |
| Duplicar filas **sin** pesos | Las predicciones cambian hasta 0.17 BCS, lo que justifica la ponderación |
| Peso por foto | Original + 3 variantes con peso 1/4 cada una; total = 1 en cada foto |
| Estandarización ponderada | Media ≤ 7 × 10⁻¹⁶ y varianza = 1 ± 3 × 10⁻¹⁵ |
| α | El de `select_alpha` canónico con las filas **originales** del train de cada fold; recalculado en los 87 folds: 0 diferencias |
| Particiones | Recalculadas: 0 diferencias en las vacas de test |
| Reproducibilidad | Las predicciones de A0–A6 se recalcularon con el código final: diferencia 0.0 |
| Flip y muslos | IoU ≥ 0.99 entre la máscara volteada con intercambio y la del segmentador sobre la foto volteada (20 fotos); sin intercambio, IoU de los muslos = 0.0 |
| Caché canónico | Hash idéntico antes y después de todas las etapas |
| Fugas (etapa C) | Verificadas en 37 folds externos y 1332 internos: la vaca de test nunca entra en ningún train ni en la selección de α |

**Ridge ponderado.** Minimiza `Σ_j v_j (y_j − b − a_j·w)² + α‖w‖²` con
estandarización ponderada. Como cada foto pesa 1, la escala de la pérdida frente a α es
la misma que en A0. Esto evita que multiplicar filas reduzca la regularización
efectiva.

## 5. Resultados de las etapas A y B

Tabla completa (`work/aug/final/table_full.md`). Δ = política − A0; los valores
negativos indican menor error.

| Esquema | Política | MAE (foto) [IC 95 %] | MAE (vaca) [IC 95 %] | RMSE | R² foto | R² vaca | Bias | ≤ ±0.5 | ΔMAE foto [IC 95 %] | ΔMAE vaca [IC 95 %] | Vacas mejor/peor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LOCO | A0 | 0.583 [0.474, 0.693] | 0.545 [0.465, 0.633] | 0.682 | 0.55 | 0.52 | +0.091 | 56 % | — | — | — |
| LOCO | A0-DUP | 0.583 [0.474, 0.693] | 0.545 [0.465, 0.633] | 0.682 | 0.55 | 0.52 | +0.091 | 56 % | +0.000 [−0.000, +0.000] | −0.000 [−0.000, +0.000] | 0/0 |
| LOCO | A1 | 0.576 [0.464, 0.689] | 0.539 [0.452, 0.633] | 0.677 | 0.56 | 0.52 | +0.095 | 55 % | −0.007 [−0.021, +0.007] | −0.007 [−0.025, +0.011] | 19/18 |
| LOCO | A2 | 0.590 [0.479, 0.704] | 0.549 [0.467, 0.637] | 0.691 | 0.54 | 0.51 | +0.098 | 55 % | +0.007 [−0.001, +0.016] | +0.003 [−0.001, +0.008] | 16/21 |
| LOCO | A3 | 0.609 [0.504, 0.718] | 0.566 [0.482, 0.655] | 0.701 | 0.52 | 0.48 | +0.091 | 52 % | +0.026 [+0.008, +0.047] | +0.021 [+0.006, +0.035] | 13/24 |
| LOCO | A4 | 0.598 [0.495, 0.703] | 0.558 [0.472, 0.649] | 0.691 | 0.54 | 0.49 | +0.080 | 53 % | +0.016 [−0.009, +0.044] | +0.013 [−0.007, +0.034] | 17/20 |
| LOCO | A5 | 0.599 [0.487, 0.714] | 0.563 [0.472, 0.662] | 0.701 | 0.52 | 0.47 | +0.095 | 53 % | +0.016 [−0.003, +0.033] | +0.018 [−0.011, +0.048] | 13/24 |
| LOCO | A6 | 0.601 [0.488, 0.720] | 0.565 [0.474, 0.663] | 0.702 | 0.52 | 0.47 | +0.097 | 53 % | +0.018 [−0.003, +0.038] | +0.019 [−0.011, +0.051] | 14/23 |
| LOCO | B-BR05 | 0.588 [0.478, 0.702] | 0.549 [0.467, 0.637] | 0.689 | 0.54 | 0.51 | +0.097 | 56 % | +0.006 [−0.001, +0.012] | +0.004 [−0.000, +0.007] | 16/21 |
| LOCO | B-BR10 | 0.589 [0.477, 0.704] | 0.548 [0.466, 0.637] | 0.690 | 0.54 | 0.51 | +0.098 | 55 % | +0.007 [−0.001, +0.015] | +0.003 [−0.001, +0.007] | 15/22 |
| LOCO | B-CT05 | 0.589 [0.481, 0.699] | 0.548 [0.467, 0.637] | 0.687 | 0.54 | 0.51 | +0.095 | 55 % | +0.006 [+0.001, +0.012] | +0.003 [+0.000, +0.007] | 16/21 |
| LOCO | B-CT10 | 0.588 [0.480, 0.699] | 0.546 [0.465, 0.633] | 0.686 | 0.54 | 0.52 | +0.097 | 56 % | +0.005 [−0.001, +0.012] | +0.001 [−0.003, +0.005] | 23/14 |
| 5×10 | A0 | 0.629 [0.501, 0.759] | 0.571 [0.482, 0.665] | 0.729 | 0.48 | 0.46 | +0.119 | 50 % | — | — | — |
| 5×10 | A0-DUP | 0.629 [0.501, 0.759] | 0.571 [0.482, 0.665] | 0.729 | 0.48 | 0.46 | +0.119 | 50 % | −0.000 [−0.000, +0.000] | −0.000 [−0.000, +0.000] | 0/0 |
| 5×10 | A1 | 0.623 [0.494, 0.755] | 0.566 [0.474, 0.664] | 0.724 | 0.49 | 0.46 | +0.121 | 50 % | −0.005 [−0.018, +0.007] | −0.005 [−0.021, +0.010] | 22/15 |
| 5×10 | A2 | 0.640 [0.509, 0.776] | 0.575 [0.484, 0.672] | 0.741 | 0.47 | 0.45 | +0.124 | 50 % | +0.011 [+0.000, +0.023] | +0.005 [−0.001, +0.010] | 12/25 |
| 5×10 | A3 | 0.649 [0.514, 0.788] | 0.581 [0.488, 0.680] | 0.748 | 0.46 | 0.43 | +0.124 | 50 % | +0.020 [−0.004, +0.047] | +0.010 [−0.004, +0.025] | 16/21 |
| 5×10 | A4 | 0.644 [0.511, 0.779] | 0.578 [0.486, 0.676] | 0.741 | 0.47 | 0.44 | +0.120 | 49 % | +0.016 [−0.010, +0.043] | +0.008 [−0.010, +0.026] | 21/16 |
| 5×10 | A5 | 0.645 [0.513, 0.782] | 0.583 [0.487, 0.685] | 0.747 | 0.46 | 0.42 | +0.125 | 48 % | +0.017 [−0.001, +0.035] | +0.013 [−0.013, +0.039] | 12/25 |
| 5×10 | A6 | 0.649 [0.515, 0.788] | 0.588 [0.492, 0.689] | 0.748 | 0.46 | 0.41 | +0.128 | 48 % | +0.020 [+0.002, +0.039] | +0.018 [−0.009, +0.046] | 15/22 |
| 5×10 | B-BR05 | 0.638 [0.508, 0.772] | 0.573 [0.482, 0.669] | 0.739 | 0.47 | 0.45 | +0.124 | 50 % | +0.009 [+0.000, +0.018] | +0.002 [−0.003, +0.007] | 16/21 |
| 5×10 | B-BR10 | 0.639 [0.506, 0.775] | 0.572 [0.481, 0.670] | 0.741 | 0.47 | 0.45 | +0.125 | 50 % | +0.010 [−0.001, +0.020] | +0.002 [−0.004, +0.008] | 16/21 |
| 5×10 | B-CT05 | 0.639 [0.508, 0.774] | 0.575 [0.484, 0.671] | 0.741 | 0.47 | 0.45 | +0.124 | 50 % | +0.011 [+0.002, +0.020] | +0.004 [+0.000, +0.008] | 18/19 |
| 5×10 | B-CT10 | 0.638 [0.507, 0.774] | 0.572 [0.481, 0.668] | 0.739 | 0.47 | 0.45 | +0.125 | 49 % | +0.010 [−0.000, +0.019] | +0.001 [−0.004, +0.006] | 19/18 |

Cuando un límite aparece como "+0.000" o "−0.000" es un valor menor que 0.0005 en
valor absoluto. Los 8 intervalos que excluyen 0 están en la sección 8.

**Lo que muestran los datos:**
- **Geométricas.** La rotación es la que más empeora: +0.026 en LOCO, con IC que
  excluye 0. Traslación y escala tienen la misma dirección, aunque sin significación.
  Coincide con lo reportado por Bär et al. (2024) para augmentation sobre
  representaciones congeladas.
- **Fotometría.** Sube el MAE entre +0.005 y +0.011, con efecto similar a ±5 % y a
  ±10 % (no hay relación dosis-respuesta).
- **Combinaciones.** A5 y A6 heredan el deterioro de las geométricas y pierden R² por
  vaca: 0.46 → 0.41–0.42 en 5×10.
- **Bias.** El sesgo positivo (+0.08 a +0.13; el modelo sobreestima en promedio) es
  propio del modelo canónico y ninguna política lo corrige.

## 6. Estabilidad del flip horizontal (A1) entre semillas

A1 se repitió con 10 semillas de augmentation adicionales (s1–s10, más la original s0),
con las mismas particiones y el mismo α.

| Esquema | Métrica | Media Δ | SD entre semillas | Rango | Δ < 0 | IC que excluye 0 |
|---|---|---|---|---|---|---|
| LOCO | MAE por foto | −0.0077 | 0.0023 | −0.0127 a −0.0034 | 11/11 | 0/11 |
| LOCO | MAE por vaca | −0.0057 | 0.0018 | −0.0100 a −0.0032 | 11/11 | 0/11 |
| 5×10 | MAE por foto | −0.0076 | 0.0024 | −0.0112 a −0.0032 | 11/11 | 0/11 |
| 5×10 | MAE por vaca | −0.0052 | 0.0012 | −0.0078 a −0.0036 | 11/11 | 0/11 |

**Interpretación:**
- El signo del efecto no depende del azar de la augmentation: Δ < 0 en las 11 semillas
  y en ambos esquemas.
- Pero las 11 repeticiones usan las **mismas 37 vacas** y no son observaciones
  independientes. La dispersión entre semillas (SD ≈ 0.002) es un orden de magnitud
  menor que la incertidumbre por animal (IC de ≈ 0.03 de ancho).
- Esto demuestra **reproducibilidad numérica**, no generalización a otras vacas.

## 7. Validación anidada por vaca (etapa C)

**Pregunta:** ¿un procedimiento que elige entre A0 y A1 usando solo vacas de
entrenamiento supera a usar siempre A0?

**Diseño**, fijado en un commit antes de producir resultados (`51c00b9`):
- externo: LOCO, con los mismos 37 folds;
- interno: LOCO sobre las 36 vacas de entrenamiento;
- α canónico calculado solo con el train interno;
- criterio: MAE promedio por vaca interno;
- regla primaria: A1 si mejora a A0 por más de δ = 0;
- sensibilidad: δ = 0.01.

| Procedimiento | Folds A0 / A1 | MAE foto | MAE vaca | RMSE | R² foto / vaca | ΔMAE foto vs A0 [IC 95 %] | ΔMAE vaca vs A0 [IC 95 %] |
|---|---|---|---|---|---|---|---|
| A0 canónico | — | 0.583 | 0.545 | 0.682 | 0.549 / 0.519 | — | — |
| Anidado, δ = 0 (primario) | 0 / 37 | 0.576 | 0.539 | 0.678 | 0.555 / 0.517 | −0.007 [−0.021, +0.007] | −0.007 [−0.025, +0.011] |
| Anidado, δ = 0.01 (sensibilidad) | 31 / 6 | 0.589 | 0.553 | 0.689 | 0.540 / 0.503 | +0.006 [+0.001, +0.013] | +0.008 [+0.001, +0.016] |

- **Regla primaria.** Eligió A1 en todos los folds, porque su ventaja interna fue
  siempre negativa (de −0.0112 a −0.0008). Por eso equivale a "A1 siempre", y su
  mejora tampoco es concluyente.
- **Regla con δ = 0.01.** Eligió A1 en las 6 vacas donde A1 empeora (0/6 mejoran) y
  resultó peor que A0. Es un artefacto estructural del LOCO con efectos pequeños: la
  ventaja interna de A1 se correlaciona con su efecto en la vaca excluida con
  **r = −0.58**. Excluir del conjunto interno una vaca perjudicada por A1 aumenta la
  ventaja media de A1 en las restantes.
- **Detalle por vaca.** La tabla con política elegida, predicción y error por vaca
  está en `work/aug/nested_A0A1/per_cow.csv` y `report.md`.
- **Qué evalúa esta etapa.** El **procedimiento de selección**, no confirma de forma
  independiente una mejora. A1 se eligió como candidato después de ver las etapas A y B
  en las mismas vacas, y la validación anidada no elimina ese sesgo.

## 8. Diferencias pareadas e intervalos por vaca

**Tabla compacta para publicación:** `work/aug/final/table_compact.{md,csv,tex}`.

| Policy | Description | LOCO MAE | LOCO ΔMAE vs A0 [95 % CI] | 5-fold×10 MAE | 5-fold×10 ΔMAE vs A0 [95 % CI] | Assessment |
|---|---|---|---|---|---|---|
| A0 | No augmentation (baseline) | 0.583 | — | 0.629 | — | Reference |
| A1 | Horizontal flip | 0.576 | −0.007 [−0.021, +0.007] | 0.623 | −0.005 [−0.018, +0.007] | Lower MAE, n.s. |
| A3 | Rotation ±5° | 0.609 | +0.026 [+0.008, +0.047] | 0.649 | +0.020 [−0.004, +0.047] | Higher MAE† |
| A4 | Translation ±3 %, scale ±3 % | 0.598 | +0.016 [−0.009, +0.044] | 0.644 | +0.016 [−0.010, +0.043] | Higher MAE, n.s. |
| B-BR05 | Brightness ±5 % | 0.588 | +0.006 [−0.001, +0.012] | 0.638 | +0.009 [+0.000, +0.018] | Higher MAE† |
| B-BR10 | Brightness ±10 % | 0.589 | +0.007 [−0.001, +0.015] | 0.639 | +0.010 [−0.001, +0.020] | Higher MAE, n.s. |
| B-CT05 | Contrast ±5 % | 0.589 | +0.006 [+0.001, +0.012] | 0.639 | +0.011 [+0.002, +0.020] | Higher MAE† |
| B-CT10 | Contrast ±10 % | 0.588 | +0.005 [−0.001, +0.012] | 0.638 | +0.010 [−0.000, +0.019] | Higher MAE, n.s. |
| A2 | Brightness + contrast ±10 % | 0.590 | +0.007 [−0.001, +0.016] | 0.640 | +0.011 [+0.000, +0.023] | Higher MAE† |
| A5 | A1 + A2 + A3 + A4 | 0.599 | +0.016 [−0.003, +0.033] | 0.645 | +0.017 [−0.001, +0.035] | Higher MAE, n.s. |
| A6 | A5 + Gaussian noise | 0.601 | +0.018 [−0.003, +0.038] | 0.649 | +0.020 [+0.002, +0.039] | Higher MAE† |
| Nested | A0/A1 chosen on training cows (δ = 0; A1 in 37/37 folds) | 0.576 | −0.007 [−0.021, +0.007] | — | — | Lower MAE, n.s. |

n.s.: el IC 95 % incluye 0 en los cuatro análisis (MAE por foto y por vaca × LOCO y
5×10). †: el IC excluye 0 en al menos uno de los cuatro análisis, sin ajuste por
multiplicidad.

**Figura** (`work/aug/final/figure_ablation.{pdf,png}`, 300 dpi). Muestra el ΔMAE por
foto con su IC 95 % por política y esquema, las estimaciones de A1 con las otras 10
semillas y la selección anidada.

**Intervalos que excluyen 0** (8 de 40; los 8 indican mayor error, ninguno menor):
- A3 en LOCO, por foto y por vaca;
- B-CT05 en LOCO por foto y en 5×10 por foto y por vaca;
- A2, A6 y B-BR05 en 5×10 por foto.

## 9. Limitaciones

1. **Tamaño muestral.**
   - La unidad independiente es la vaca, y solo hay 37.
   - Las diferencias de ±0.02 BCS están dentro del ruido.
   - El ancho de los IC del ΔMAE por vaca depende de la política: ≈ 0.01 en las
     fotométricas, ≈ 0.03–0.04 en el flip y hasta ≈ 0.06 en las combinadas. En el flip
     el intervalo es unas 5 veces mayor que el efecto estimado.
   - El 5×10 promedia repeticiones sobre las mismas vacas y no agrega animales.
2. **Comparaciones múltiples.**
   - Se evaluaron 10 políticas × 4 análisis = 40 intervalos sin ajuste por
     multiplicidad.
   - Los análisis están correlacionados (mismas vacas y políticas anidadas), así que no
     se puede derivar un número esperado de falsos positivos con exactitud.
   - Que los 8 intervalos significativos apunten en la misma dirección (más error)
     sugiere un patrón, pero ningún resultado individual debe presentarse como
     confirmatorio.
3. **Selección exploratoria de políticas.**
   - Las intensidades no se optimizaron.
   - La etapa B y la etapa C se diseñaron después de ver la etapa A; la elección de A1
     como candidato es post hoc.
   - Las conclusiones valen para estas magnitudes.
4. **Dependencia del pipeline.**
   - La augmentation solo afecta a un ridge lineal sobre descriptores congelados.
   - Los resultados no se extienden a modelos que ajustan el backbone, donde la
     augmentation puede actuar de otro modo.
   - No se evaluó TTA en el descriptor; se excluyó por diseño.
5. **Máscaras.**
   - Las variantes reutilizan las máscaras canónicas transformadas en lugar de
     resegmentar cada imagen aumentada.
   - Esto se verificó para el flip (IoU ≥ 0.99), pero no para rotación ni escala.
6. **Etiquetas.**
   - Hay un solo BCS por vaca en 36 de 37 casos, con resolución de 0.25 punto.
   - No hay medidas de concordancia entre evaluadores en este conjunto de datos.
7. **Validación externa.**
   - Todas las vacas provienen de un mismo hato y robot.
   - No existe un conjunto independiente de vacas nuevas para confirmar ningún efecto.

## 10. Conclusiones para el artículo

1. **Ninguna augmentation mejoró de forma estadísticamente respaldada.** En un modelo
   de BCS con descriptores regionales DINOv2-L congelados y regresión ridge, evaluado
   con validación por vaca (37 vacas, 369 fotos), ninguna de las diez políticas
   convencionales redujo el error de predicción.
2. **El flip horizontal es reproducible, no demostrado.** Fue la única política con
   menor error en todos los análisis (ΔMAE de −0.005 a −0.007 BCS, ≈ 1 % del MAE), de
   forma reproducible en 11 semillas de augmentation. Sus IC 95 % por vaca incluyeron 0
   en todos los casos. Debe describirse como una **pequeña mejora numérica
   reproducible**, no como una mejora estadísticamente respaldada.
3. **Las transformaciones geométricas aumentaron el error.** La rotación de ±5° lo hizo
   con IC que excluye 0 en LOCO. Esto coincide con lo reportado para augmentation sobre
   representaciones congeladas de transformers visuales (Bär et al., 2024).
4. **La fotometría leve no aportó beneficio.** Brillo y contraste de ±5–10 % aumentaron
   ligeramente el error, sin relación dosis-respuesta.
5. **La selección anidada no superó al modelo sin augmentation.** La selección A0/A1
   con validación anidada por vaca eligió el flip en todos los folds, sin mejora
   concluyente. Con efectos tan pequeños, las reglas con umbral en LOCO son inestables
   por la correlación negativa entre la ventaja interna y el efecto externo.
6. **Recomendación.** El modelo sin augmentation sigue siendo el modelo principal. Con
   cámara y encuadre fijos, la variabilidad que introduce la augmentation convencional
   no parece compensar la que ya captura un descriptor preentrenado congelado.
   Confirmar o descartar el efecto del flip requiere vacas no usadas en ninguna etapa
   del estudio.

**Texto sugerido para Resultados (inglés):**

> Training-time augmentation did not improve BCS prediction. Relative to the
> unaugmented model (leave-one-cow-out MAE 0.583), horizontal flipping yielded a small,
> seed-stable reduction in MAE (−0.007; 95 % CI −0.021 to 0.007; 11 of 11 augmentation
> seeds negative), whereas rotation (±5°) increased the error (+0.026; 95 % CI 0.008 to
> 0.047). Photometric and combined policies increased MAE by 0.005–0.020 BCS units.
> A nested cow-level procedure selecting between no augmentation and horizontal flipping
> did not outperform the unaugmented model.

**Referencias generales** que el texto menciona y que **no se verificaron en este
estudio**; hay que confirmarlas antes de enviar el artículo:
- Oquab et al. (2024), DINOv2, *TMLR*;
- Bishop (1995), *Neural Computation* 7:108–116 (ruido como regularización de Tikhonov);
- Cawley y Talbot (2010), *JMLR* 11:2079–2107, y Varma y Simon (2006), *BMC
  Bioinformatics* 7:91 (sesgo de selección y validación anidada);
- Holm (1979), *Scand. J. Stat.* 6:65–70;
- Perez et al. (2018), FiLM, *AAAI*.

---

## Anexo: propuesta técnica M0 / M1 / M2 (no implementada)

### A.1 Objetivo

Comparar, con la misma evaluación por vaca:
- **M0:** DINOv2 + ridge (modelo canónico, A0);
- **M1:** DINOv2 + DEL + ridge;
- **M2:** DINOv2 + FiLM condicionado por el DEL.

No se modifican `Dataset_Fr_BCS`, las particiones (`splits.json`, LOCO y 5×10) ni el
modelo canónico. Ningún modelo usa augmentation (se emplean los descriptores A0).

### A.2 Datos de DEL y antecedentes

- **Fuente.** La rama `origin/model/del` (commit `8bacd31`) calcula el DEL por foto a
  partir de los exportes diarios del robot (`GeneralidadesOrdeno/`). Cada foto se cruza
  por vaca y fecha con el exporte más cercano, con un máximo de 7 días, y el resultado
  queda en `work/del.csv`. Según esa rama, solo 1 de las 369 fotos queda sin DEL.
- **Uso.** El DEL es una **covariable auxiliar** unida por nombre de foto; no altera el
  dataset ni las particiones. Antes de empezar se congela `del.csv` y se registra su
  hash.
- **Faltantes.** Se imputan con la mediana del train de cada fold.
- **Antecedentes reportados en esa rama** (no reproducidos aquí):
  - DEL lineal: MAE LOCO de 0.616; DEL con spline: 0.634; frente a 0.583 de M0.
  - En 5×10: 0.630 y 0.632, frente a 0.629.
  - Ninguna diferencia es significativa.
  - Un FiLM sobre ConvNeXt (rama `Adri`) dio MAE 0.81 contra 0.82.
- **Problema de identificabilidad.**
  - 36 de 37 vacas tienen un único BCS, así que el DEL varía dentro de la vaca pero la
    etiqueta no. Su efecto entre vacas se confunde con la identidad de la vaca.
  - La relación DEL–BCS es en forma de U y hay muy pocas vacas por tramo de lactancia.
  - Esto limita lo que M1 y M2 pueden aprender, y cualquier diseño debe controlarlo.

### A.3 Modelos

Sean `z ∈ R^6150` el descriptor canónico y `φ(d) ∈ R^q` una base del DEL. La base es el
DEL estandarizado (q = 1) o un B-spline cúbico con nodos fijados en el train de cada
fold (q ≈ 4–6).

- **M0:** `ŷ = wᵀz + b`, el ridge canónico sin cambios.
- **M1:** ridge sobre `[z, φ(d)]` con **penalización por bloques**:
  - α para el bloque de imagen y α_D para el bloque DEL (ridge generalizado, forma
    cerrada).
  - Equivale al "peso w" de la rama `model/del`, pero explícito.
  - Sin ese bloque, el DEL quedaría anulado por la penalización común de las 6150
    columnas.
- **M2:** FiLM por región: `z̃_r = γ_r(d) ⊙ z_r + β_r(d)` y luego `ŷ = wᵀz̃ + b`, donde
  `γ_r(d) = 1 + A_r φ(d)` y `β_r(d) = B_r φ(d)`.
  - **M2-lin (principal).** Como la cabeza es lineal, `wᵀz̃` se expande en un término
    de imagen (`z`), interacciones `φ_k(d)·z` y un término DEL (`φ(d)`). Por eso FiLM
    con cabeza lineal equivale a un ridge sobre `[z, φ(d) ⊗ z, φ(d)]` con restricción
    de rango bajo. Se propone la versión relajada y convexa: ridge con tres bloques de
    penalización (α, α_int, α_D) y solución cerrada en forma dual, con la misma
    maquinaria que el modelo canónico. Es determinista y no requiere entrenamiento por
    gradiente.
  - **M2-nn (secundario).** FiLM entrenado por gradiente sobre descriptores reducidos
    por PCA por región, ajustada en el train de cada fold (p. ej. 32–64 componentes).
    γ y β se generan con una red muy pequeña o con la misma base spline. Lleva weight
    decay, early stopping en la validación interna y promedio de varias semillas de
    inicialización. Solo se justifica si M2-lin muestra señal.

### A.4 Control de la capacidad adicional

1. **Todos los hiperparámetros se eligen dentro del train**, con `GroupKFold` por vaca
   y rejillas fijadas de antemano: α, α_D, α_int, nodos, dimensión PCA, tasa de
   aprendizaje, weight decay y épocas. α se selecciona con el mismo criterio que
   `select_alpha`.
2. **Capacidad efectiva reportada.** Se reportan los grados de libertad efectivos
   `tr(H)` del ridge por bloques en cada fold. Si M2-lin solo mejora con más grados de
   libertad, eso se hace explícito.
3. **Controles de igual capacidad:**
   - **M1-perm / M2-perm:** se permutan entre vacas las trayectorias de DEL (cada vaca
     conserva la estructura de sus fotos), repetido R = 50 veces. Da una distribución
     nula de ΔMAE con la misma capacidad y sin información real del DEL.
   - **M2-const:** igual a M2, pero con γ y β constantes (no dependen del DEL). Separa
     "más parámetros" de "condicionamiento por DEL".
   - **DEL solo:** ridge sobre `φ(d)`. Cuantifica cuánto BCS explica el DEL por sí solo.
4. **Selección anidada** si hay que elegir entre variantes (p. ej. lineal frente a
   spline, o M2-lin frente a M2-nn). Se usa el mismo esquema de la etapa C: el error
   reportado es el del procedimiento y no el de la mejor variante.

### A.5 Evaluación comparable

- **Particiones:** las mismas `splits.json` para M0, M1, M2 y los controles; mismas fotos
  y mismas vacas de test.
- **Métricas:** las mismas que en este informe, con ΔMAE pareado contra M0 e IC 95 %
  por bootstrap de vacas.
- **Análisis por vaca:** además del MAE por foto, se reporta el MAE promedio por vaca
  como métrica principal, porque el DEL solo puede aprender diferencias entre vacas.
- **Pre-registro** en un commit antes de correr:
  - comparación primaria: M2-lin frente a M0, MAE por vaca en LOCO;
  - comparaciones secundarias: M1 frente a M0, M2 frente a M1 y el 5×10;
  - corrección de Holm para las secundarias.
- **Criterio de mejora**, fijado antes de correr. Debe cumplir las tres condiciones:
  1. IC 95 % del ΔMAE por vaca < 0 en LOCO y la misma dirección en 5×10;
  2. ΔMAE por debajo del percentil 5 de la distribución nula por permutación;
  3. |ΔMAE| ≥ 0.02 BCS, que es el umbral de relevancia práctica usado en el proyecto.
- **Reproducibilidad:** hash de `features.npz` y `del.csv`, semillas y versiones en el
  `config.json` de cada modelo. Los resultados se guardan en una carpeta nueva
  (`work/del_film/`).

### A.6 Riesgos previstos

- Dados los antecedentes de la rama `model/del` y la falta de variación del BCS dentro
  de cada vaca, el resultado más probable es nulo.
- El experimento sigue siendo útil para el artículo si se controla la capacidad: un
  nulo bien controlado ("el DEL no aporta información más allá de la imagen con este
  diseño") es publicable.
- La única forma de identificar el efecto del DEL dentro de la vaca es contar con
  vacas con varios BCS a lo largo de la lactancia.

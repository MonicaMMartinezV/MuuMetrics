# Ablación de data augmentation (rama `model/dataaug`)

Estudio de qué augmentation de entrenamiento mejora el BCS estimado en vacas no vistas,
sobre el modelo canónico **sin modificarlo**. `bcs_regions.py`, `segmenter.py` y
`data.py` no cambian, y los comandos del README se comportan igual.

| Archivo | Qué hace |
|---|---|
| `augment.py` | Políticas A0–A6 y A0-DUP, sorteo reproducible y transformación coherente de foto + 6 máscaras |
| `ablation.py` | Descriptores aumentados, ridge ponderado, controles, evaluación, tabla y gráficos |

## 1. Diseño

**Qué se aumenta.** Solo el train del ridge. Por cada foto original se agregan K = 3
variantes. Su descriptor se calcula con el mismo `describe()` y el mismo DINOv2-L
congelado (378 px, misma normalización). Las máscaras son las del caché canónico
(`work/masks`, producidas por el segmentador con su TTA de flip sin cambios). Las
transformaciones geométricas se aplican con el mismo `theta` a la foto y a las máscaras.
El segmentador no se vuelve a correr ni a entrenar.

**Qué no se aumenta.** El test usa siempre las filas originales de `work/features.npz`.

**Flip y muslos.** En las 369 fotos, `thigh_L` está a la izquierda de la imagen y
`thigh_R` a la derecha, así que los nombres son por lado de la imagen. Al voltear se
intercambian los canales 4 y 5 (`FLIP` de `segmenter.py`), y `thigh_L` sigue a la
izquierda. `ablation.py checkflip` lo verifica contra el segmentador: corre el
segmentador sobre la foto volteada y compara su resultado con la máscara volteada,
con y sin intercambio.

**Políticas.** Las intensidades son propuestas iniciales del equipo; no están
optimizadas ni copiadas de un artículo.

| Política | Operaciones (en cada variante) |
|---|---|
| A0 | Sin augmentation (`fit_predict` canónico) |
| A0-DUP | 3 copias idénticas de cada foto (control) |
| A1 | Flip horizontal con p = 0.5 (+ intercambio de muslos) |
| A2 | Con p = 0.5: brillo × U(0.9, 1.1) y contraste × U(0.9, 1.1), estilo ColorJitter |
| A3 | Rotación U(−5°, 5°) |
| A4 | Traslación U(−3 %, 3 %) en x e y, escala U(0.97, 1.03) |
| A5 | A1 + A2 + A3 + A4 |
| A6 | A5 + ruido gaussiano σ = 0.02 (en escala 0–1, ≈ 5/255) |

- Orden de aplicación: flip → afín → brillo/contraste → ruido.
- El relleno fuera del cuadro es el color medio ImageNet, igual que el lienzo del segmentador.
- Las máscaras se interpolan bilinealmente y `describe()` las umbraliza en 0.5, como siempre.

**Semillas.** Cada variante tiene su semilla
`SeedSequence([BASE_SEED=0, crc32(nombre de la foto), k])`. Todos los parámetros se
sortean siempre; la política solo elige cuáles usar. Por eso las políticas son
anidadas: A5 usa los mismos flips que A1 y los mismos ángulos que A3. Los parámetros
efectivos de cada variante quedan en `work/aug/features/<política>.params.csv`.

**Control de regularización del ridge.** El ridge canónico minimiza
`Σ_i (y_i − b − a_i·w)² + α‖w‖²` sobre columnas estandarizadas con la media y la sd del
train; el intercepto no se penaliza. Si se multiplican las filas por 4, la suma de
errores crece 4× frente a α: la regularización efectiva baja y las fotos con más
variantes pesan más. Por eso:

- cada fila (original y variantes) pesa `1/(K+1)`, así que cada foto pesa 1 en total,
  igual que en A0;
- el ridge ponderado minimiza `Σ_j v_j (y_j − b − a_j·w)² + α‖w‖²`;
- la estandarización es ponderada (media y sd con los mismos pesos);
- `α` de cada fold es el que el `select_alpha()` canónico elige con las filas
  **originales** del train de ese fold (`work/aug/splits.json`).

Con copias idénticas (A0-DUP) esto es algebraicamente el problema de A0. La prueba
unitaria da una diferencia de ~1e-14. Sin los pesos, las predicciones cambian hasta
0.17 BCS.

**Controles, en orden y obligatorios.** `run --policy A1 ...` se niega a correr si no
existe `work/aug/controls.json` con `ok: true`. Los controles son:

1. A0 reproduce `work/preds_loco.csv` y `work/preds_rep5x10.csv` canónicos (tolerancia 6e-5,
   porque esos CSV tienen 4 decimales).
2. A0-DUP coincide con A0 en todas las predicciones de todas las repeticiones (tolerancia 1e-6).
3. Se reporta la diferencia máxima entre los descriptores de A0-DUP, recalculados por la
   ruta de augmentation con la identidad, y el caché canónico.

**Evaluación.** Las particiones son exactamente las de `cmd_evaluate`:
`GroupKFold(37)` para LOCO y `StratifiedGroupKFold(5, shuffle, random_state=0..9)`
estratificado por BCS×2. Las métricas son las de `metrics()` canónico más:

- **Bias:** media de pred − real.
- **IC 95 % del MAE por foto:** bootstrap por vaca.
- **ΔMAE pareado contra A0:** por foto y por vaca, con IC 95 % por bootstrap de vacas
  (5000 réplicas, semilla 0, las mismas vacas remuestreadas para todas las políticas).
- **Vacas que mejoran o empeoran.**

En 5×10, como en el canónico, se promedian las predicciones de las 10 repeticiones
antes de calcular las métricas.

## 2. Comandos

Desde `Modeling/Main model/BCS-Regions/`, con el pipeline canónico ya corrido
(`work/masks`, `work/features.npz`, `work/preds_*.csv`):

```bash
# Controles (A0 y A0-DUP) y verificación del flip
docker compose run --rm --entrypoint python bcs ablation.py run --policy A0 A0-DUP
docker compose run --rm --entrypoint python bcs ablation.py checkflip --n 20

# Una sola ablación (requiere controles aprobados)
docker compose run --rm --entrypoint python bcs ablation.py run --policy A3

# Batería completa (controles primero; se detiene si fallan)
docker compose run --rm --entrypoint python bcs ablation.py run --policy all

# Solo regenerar la tabla y los gráficos
docker compose run --rm --entrypoint python bcs ablation.py report
```

Salidas (no se suben al repo; `work/` está en `.gitignore`). El caché canónico solo se lee.
Las tablas y figuras agregadas de las etapas A, B y C están publicadas en
[`results_dataaug/`](results_dataaug/README.md); las predicciones por foto, los ejemplos `qc/`
y los descriptores quedan solo en local.

```
work/aug/
├── features/<P>.npz, <P>.params.csv   descriptores aumentados [n, K, 6150] y parámetros por variante
├── splits.json                        vacas de test y alpha canónico por fold, hash de features.npz
├── controls.json, checkflip.json      controles
├── qc/<P>/                            ejemplos: original | aumentada (thigh_L rojo, thigh_R azul)
├── results/<P>/                       metrics.csv, preds_*.csv, preds_*_reps.npy, config.json
└── report/                            comparison.csv/.md, delta_mae_vs_A0.png/.pdf
```

## 3. Qué es defendible y qué no

- **No se elige la "mejor" política con las vacas de test.** Esta tabla es una
  ablación descriptiva. Si se quiere adoptar una política, hay que elegirla con
  validación interna por vaca dentro de cada fold (selección anidada) y reportar
  ese resultado anidado, no el máximo de la tabla.
- **El efecto se mide contra A0 de forma pareada** (mismas vacas, mismos folds, mismo α).
  Una diferencia cuyo IC 95 % por vaca incluye 0 no es una mejora. Con 37 vacas,
  ±0.02 de MAE está dentro del ruido.
- **Hay siete comparaciones.** Si se presenta alguna como significativa, hay que
  ajustar por multiplicidad (p. ej. Holm) o declararla exploratoria.
- **LOCO y 5×10 deben coincidir en el signo.** Un efecto que solo aparece en un esquema es frágil.
- **Las intensidades no se optimizaron.** Un resultado nulo dice "estas intensidades
  no ayudan", no "la augmentation no sirve".

## 4. Literatura

Los artículos de referencia del equipo son:
- Yao et al. (2026), doi:10.3168/jds.2025-27759
- Siachos et al. (2024), doi:10.3168/jds.2023-23894
- Swastika (2025), doi:10.14313/jamris-2025-038

**Todavía no se ha verificado** qué augmentation usa cada uno: no se pudo leer el texto
completo (ScienceDirect respondió 403 y el servidor de JAMRIS rechazó la conexión).
Las políticas A1–A6 de arriba son una adaptación del equipo a este pipeline (descriptor
congelado + ridge) y no deben presentarse como las de esos artículos hasta comparar
contra sus secciones de métodos.

---

# Etapa B: estabilidad del flip, fotometría aislada y FroFA

## B0. Auditoría previa (`ablation.py audit`, solo lectura → `work/aug/audit.json`)

| Comprobación | Resultado |
|---|---|
| Caché canónico (`features.npz`, máscaras, preds, `ridge_model.npz`) | Hash idéntico al de antes de la ablación |
| Particiones (87 folds: 37 LOCO + 50 del 5×10) recalculadas desde cero | 0 folds con vacas de test distintas a `splits.json` |
| α canónico recalculado con `select_alpha` en cada fold | 0 folds con α distinto; todas las políticas leen el mismo `splits.json` |
| Predicciones de A0–A6 recalculadas con el código actual | Diferencia máxima 0.0 en todas |
| Peso total por foto (original + 3 variantes) | Exactamente 1 en todas las fotos; peso total = nº de fotos |
| Estandarización ponderada | Media ponderada ≤ 7e-16 y varianza ponderada = 1 ± 3e-15 |
| Unidad del bootstrap | Vaca (5000 réplicas, semilla 0); en 5×10 se promedian las 10 repeticiones por foto antes de remuestrear |

A0/A0-DUP, A1–A3 y A4–A6 se corrieron con tres versiones de `ablation.py`. Solo
cambiaban el código de gráficos (caché de matplotlib y posición de la leyenda); la
reproducción exacta de todas las predicciones con el código actual lo confirma.
`augment.py`, `bcs_regions.py`, `segmenter.py` y `data.py` tienen el mismo hash en
todas las corridas.

## B1. Estabilidad de A1 con otras semillas de augmentation

Son `A1@s1` … `A1@s10`: la misma política A1 (flip con p = 0.5, K = 3) con otra semilla
de augmentation (`SeedSequence([s, crc32(foto), k])`). `A1` es la semilla 0. Las
particiones, el α y los pesos no cambian.

Las 11 semillas usan las mismas 37 vacas. Su dispersión mide cuánto depende el
resultado del sorteo de flips; **no** son muestras independientes y no se les aplica
ninguna prueba como si lo fueran. La incertidumbre por animal es la del IC 95 %
pareado por vaca de cada semilla.

## B2. Fotometría aislada

| Política | Operación (en cada variante, con p = 0.5) |
|---|---|
| B-BR05 | Brillo × U(0.95, 1.05) |
| B-BR10 | Brillo × U(0.90, 1.10) |
| B-CT05 | Contraste × U(0.95, 1.05) alrededor del gris medio |
| B-CT10 | Contraste × U(0.90, 1.10) |

- La compuerta p = 0.5 es la misma que en A1 y A2.
- Por el sorteo anidado, B-BR10 y B-CT10 usan las mismas compuertas y los mismos
  factores que A2, así que **son exactamente las dos componentes de A2**.
- No hay rotación, escala, traslación ni ruido.

## B3. Comandos

```bash
docker compose run --rm --entrypoint python bcs ablation.py audit
docker compose run --rm --entrypoint python bcs ablation.py run --policy a1seeds photometric
docker compose run --rm --entrypoint python bcs ablation.py report --report_name stageB --policy A1 a1seeds A2 photometric
```

- **Grupos:** `all` (A0–A6), `a1seeds` (A1@s1…s10) y `photometric` (B-*).
- **Otra semilla:** cualquier política acepta `@sN`.
- **Reportes:** el de la etapa A (`work/aug/report/`) no se toca; el de la etapa B se
  guarda en `work/aug/report_stageB/` (`comparison.*`, `stability_A1.md`,
  `stability_A1_summary.csv`, `stability_A1.png/.pdf`, `delta_mae_vs_A0.png/.pdf`).

## B4. Protocolo de selección anidada por vaca (propuesto, no implementado)

La tabla de ablación compara políticas con las mismas 37 vacas que se usan para
reportar su error. Elegir ahí la de menor MAE y reportar ese MAE sesga el resultado
hacia abajo, porque elige el máximo de varios estimadores con ruido. Para estimar sin
sesgo el rendimiento de **"elegir una política y usarla"**:

1. **Fijar antes de correr:**
   - los candidatos, p. ej. `{A0, A1, B-BR05, B-BR10, B-CT05, B-CT10}`;
   - el criterio: MAE promedio por vaca en la validación interna;
   - la regla de desempate: quedarse con A0 salvo que el candidato mejore el MAE
     interno en más de un umbral δ fijado de antemano, p. ej. 0.01.
2. **Externo:** el mismo esquema de reporte, LOCO y 5 folds por vaca × 10 semillas.
   Las vacas de test externas no participan en nada de lo siguiente.
3. **Interno**, solo con las vacas de train externo:
   - se usa `GroupKFold` por vaca (p. ej. 5 folds);
   - por cada candidato se entrena el ridge ponderado con α elegido por `select_alpha`
     dentro del train interno y se predicen las vacas de validación interna;
   - con el criterio se elige un candidato.
4. **Reentrenar y predecir:**
   - se reentrena el candidato elegido con todo el train externo (α canónico de ese train);
   - se predicen las vacas de test externas.
5. **Reportar:**
   - el MAE externo del **procedimiento** y su ΔMAE pareado contra A0 (IC por vaca);
   - la frecuencia con que se eligió cada candidato. Si la elección cambia mucho
     entre folds, ninguna política es robustamente mejor.

El costo es bajo: los descriptores aumentados ya están en caché y solo se reentrena el
ridge (≈ 37 × 5 × 6 ajustes en LOCO).

## B5. Frozen Feature Augmentation (FroFA): análisis de compatibilidad (no implementado)

**Fuente:** Bär, Houlsby, Dehghani, Kumar, *Frozen Feature Augmentation for Few-Shot
Image Classification*, CVPR 2024, arXiv:2403.10519. Se leyó directamente la versión HTML
de arXiv hasta la Tabla 2. Lo marcado "(no verificado)" proviene de un resumen
automático del resto del texto.

**Qué hace FroFA:**
- Cachea la salida del último bloque transformer de un ViT congelado: tokens de parche
  N×C, reacomodados como mapa √N×√N×C.
- Lleva cada valor a [0, 1] con un min-max (Ec. 5), aplica una augmentation de imagen y
  deshace el escalado (Ec. 7).
- Entrena solo la cabeza (MAP o linear probe con L2) con esos tokens aumentados.

**Variantes:**
- Por defecto: un valor aleatorio por muestra y min/max globales.
- cFroFA: un valor aleatorio por canal.
- c²FroFA: valor y min/max por canal.

**Hallazgos de la Tabla 2** (L/16 JFT-3B, ImageNet, 1 y 5 shots):
- Brillo, posterize y contraste dan mejoras consistentes, de +4.8/+3.7/+2.8 puntos
  en 1-shot.
- Las geométricas (rotate, shear, translate) siempre empeoran.
- Sobre el linear probe con L2, la mejora reportada en ImageNet 5-shot es de 0.8 puntos.
- Se evaluaron ViT Ti/B/L preentrenados con JFT-3B, ImageNet-21k y WebLI (SigLIP).
  DINOv2 no aparece entre los backbones evaluados (no verificado).
- Los mejores hiperparámetros reportados (brillo c²FroFA con z ~ U(−v, v),
  v ≈ 0.8–1.0) quedan como no verificados.

**¿Es compatible con nuestro pipeline? Sí, sin tocar la arquitectura:**
- **Siguen congelados:** el segmentador (y su TTA), DINOv2-L, las máscaras, las 6
  regiones y el promedio por región de `describe()`.
- **No cambia:** el ridge. Las filas FroFA entran como variantes con el mismo peso
  1/(K+1), el α canónico y las mismas particiones.
- **Habría que agregar:**
  1. Un paso que guarde el mapa de tokens 27×27×1024 de cada foto (≈ 0.55 GB en fp16
     para 369 fotos), o al menos el min/max global y por canal de cada foto.
  2. La función FroFA sobre ese mapa.
  3. Volver a aplicar el **mismo** promedio por región que usa `describe()`.

**La observación clave para un modelo lineal.** El promedio por región es lineal. Para
brillo y contraste (operaciones afines puntuales) y sin recorte a [0, 1]:
`promedio_región(a·f + b) = a·promedio_región(f) + b`.
- Brillo FroFA por defecto suma `z·(f_max − f_min)` a todas las componentes de los
  6 descriptores.
- cFroFA y c²FroFA suman un vector por canal, el mismo para las 6 regiones.

O sea, **sin recorte, FroFA puntual sobre tokens es exactamente una inyección de ruido
estructurado en los descriptores por región**. Se puede calcular sobre los vectores
6×1024 ya cacheados (solo hace falta el min/max por foto). Solo el recorte en [0, 1] y
posterize, que no son lineales, necesitan el mapa de tokens completo.

**Consecuencias para el ridge:**
- Entrenar un modelo lineal con ruido aditivo en la entrada equivale aproximadamente a
  una penalización de Tikhonov `wᵀ Σ_δ w` (Bishop, 1995), donde `Σ_δ` es la covarianza
  del ruido. FroFA sobre ridge funcionaría, en la práctica, como un **regularizador
  adicional y estructurado**: penaliza la dirección "brillo del mapa de tokens", común a
  las 6 regiones.
- Eso choca con la regla de no introducir regularizadores nuevos. Además interactúa con
  α, que se eligió sin augmentation.
- Si se implementa, hay que declararlo así. El control mínimo es una variante con ruido
  isotrópico de la misma energía, para separar "dirección específica" de "más
  regularización".

**Expectativa (no es un resultado):**
- Lo que FroFA confirma es coherente con lo observado en la etapa A: lo geométrico
  empeora.
- Sus mejores resultados son en 1–5 shots de clasificación con una cabeza entrenada
  por SGD. Aquí hay 37 vacas, regresión y ridge de forma cerrada con α ya validado.
  Por eso el margen esperado es pequeño, y no se debe asumir que se trasladan sus
  ganancias.

## B6. Resultados (corrida del 2026-11-09; tablas completas en [`results_dataaug/report_stageB/`](results_dataaug/report_stageB/), local `work/aug/report_stageB/`)

ΔMAE = política − A0. Valores negativos = mejor que A0. Los IC 95 % son pareados con
bootstrap por vaca.

**Estabilidad de A1 entre 11 semillas** (s0 = A1 original, s1–s10 nuevas):

| Esquema | Métrica | Media Δ | SD entre semillas | Rango | Semillas con Δ < 0 | Semillas cuyo IC excluye 0 |
|---|---|---|---|---|---|---|
| LOCO | MAE por foto | −0.0077 | 0.0023 | −0.0127 … −0.0034 | 11/11 | 0 |
| LOCO | MAE por vaca | −0.0057 | 0.0018 | −0.0100 … −0.0032 | 11/11 | 0 |
| 5×10 | MAE por foto | −0.0076 | 0.0024 | −0.0112 … −0.0032 | 11/11 | 0 |
| 5×10 | MAE por vaca | −0.0052 | 0.0012 | −0.0078 … −0.0036 | 11/11 | 0 |

**Fotometría aislada:**

| Política | LOCO ΔMAE fotos | LOCO ΔMAE vaca | 5×10 ΔMAE fotos | 5×10 ΔMAE vaca |
|---|---|---|---|---|
| B-BR05 | +0.006 [−0.001, +0.012] | +0.003 [−0.000, +0.007] | +0.009 [+0.000, +0.018] | +0.002 [−0.003, +0.008] |
| B-BR10 | +0.007 [−0.001, +0.015] | +0.003 [−0.001, +0.007] | +0.010 [−0.001, +0.020] | +0.002 [−0.004, +0.008] |
| B-CT05 | +0.006 [+0.001, +0.012] | +0.003 [+0.000, +0.007] | +0.011 [+0.002, +0.020] | +0.004 [+0.000, +0.008] |
| B-CT10 | +0.005 [−0.001, +0.012] | +0.001 [−0.003, +0.006] | +0.010 [−0.000, +0.019] | +0.001 [−0.004, +0.007] |
| A2 (referencia) | +0.007 [−0.001, +0.016] | +0.003 [−0.001, +0.008] | +0.011 [+0.000, +0.023] | +0.005 [−0.001, +0.010] |

**Lectura:**

- **Flip (A1).** La dirección es estable frente al azar de la augmentation: Δ < 0 en las
  11 semillas y en ambos esquemas. Pero:
  - la magnitud (≈ −0.006 a −0.008 BCS, ≈ 1 % del MAE) está muy por debajo del umbral
    de ruido de ±0.02;
  - ningún IC por vaca excluye 0;
  - la dispersión entre semillas (SD ≈ 0.002) es mucho menor que la incertidumbre por
    animales (ancho del IC ≈ 0.03).

  Las semillas comparten las mismas 37 vacas, así que su consistencia **no** es
  evidencia de que la mejora generalice a otras vacas. Es una tendencia reproducible,
  pero no una mejora demostrada.
- **Fotometría.** Ninguna mejora. Las cuatro tienen Δ > 0 en ambos esquemas, de
  magnitud parecida a ±5 % y a ±10 % (no hay relación dosis–respuesta). B-CT05 tiene un
  IC por foto que excluye 0 en ambos esquemas. El efecto es diminuto (+0.006 a +0.011) y
  no está corregido por multiplicidad.
- **Sin selección.** No se elige ninguna política con esta tabla. Para adoptar el flip
  hay que usar el protocolo anidado de B4.

---

# Etapa C: selección anidada por vaca entre A0 y A1 (exploratoria)

**Pregunta.** ¿Un **procedimiento** que elige entre A0 y A1 usando solo vacas de
entrenamiento predice mejor que usar siempre A0?

**Limitación declarada de antemano.** A1 se eligió como candidato *después* de ver las
etapas A y B en estas mismas 37 vacas. La validación anidada elimina el sesgo de elegir
entre A0 y A1 dentro de cada fold, pero **no** elimina el sesgo de haber elegido a A1
como candidato. Por eso esta etapa es exploratoria. Una confirmación independiente
requiere vacas nuevas, que no se hayan usado en ninguna etapa.

## Diseño (fijado en este commit, antes de producir resultados)

| Elemento | Definición |
|---|---|
| Externo | LOCO: los mismos 37 folds de `cmd_evaluate` (`GroupKFold(37)`) |
| Interno | LOCO sobre las 36 vacas de train externo (36 folds; 1332 en total). Es determinista y predice cada vaca de train una sola vez |
| Candidatos | A0 (`fit_predict` canónico) y A1 (flip p = 0.5, K = 3, semilla 0, mismas variantes de `work/aug/features/A1.npz`, ridge ponderado con peso 1/(K+1)) |
| α interno | `select_alpha` canónico con las filas originales de las 35 vacas de train interno; no ve la vaca de validación interna ni la de test externo. A0 y A1 comparten ese α |
| Criterio | MAE promedio por vaca sobre las 36 vacas de validación interna |
| Regla primaria | A1 si `MAE_vaca(A1) < MAE_vaca(A0) − δ` con **δ = 0**; empate → A0 |
| Sensibilidad | La misma regla con **δ = 0.01** (umbral propuesto en B4) |
| Ajuste final | Política elegida reentrenada con las 36 vacas, con α = `select_alpha` sobre ese train (el α canónico del fold), y una predicción por foto de la vaca excluida |
| Comparación | A0 canónico (`work/aug/results/A0`, idéntico a `work/preds_loco.csv`) y "A1 siempre" como referencia |
| Métricas | MAE por foto y por vaca, RMSE, R², bias, ≤0.5, ΔMAE pareado vs A0 con IC 95 % por bootstrap de vacas (5000, semilla 0) |

**Verificación de fugas** (`nested.py check` → `leakage_check.json`):
- en cada fold externo, la vaca de test no está en el train;
- los folds externos coinciden con el LOCO canónico;
- en los 1332 folds internos, la vaca de test externa nunca aparece;
- train y validación internos son disjuntos y juntos cubren exactamente el train externo;
- las variantes A1 están alineadas por nombre con el caché canónico, y solo se usan
  las de fotos que están en el train del ajuste.

La predicción final de la vaca externa no se usa para elegir nada.

**Controles integrados:**
- el α final de cada fold debe coincidir con el de `splits.json`;
- las predicciones finales de A0 y A1 deben reproducir `results/A0` y `results/A1`.

```bash
docker compose run --rm --entrypoint python bcs nested.py check
docker compose run --rm --entrypoint python bcs nested.py run      # -> work/aug/nested_A0A1/
```

Salidas en `work/aug/nested_A0A1/`:
- `prereg.json`: diseño, fecha y hashes, escrito antes de ajustar;
- `leakage_check.json`;
- `folds.json`: selección, MAE interno y predicciones por fold;
- `comparison.csv`, `per_cow.csv`, `preds_nested_loco.csv`, `report.md` y `summary.json`.

## Resultados de la etapa C (corrida del 2026-11-09; agregados en [`results_dataaug/nested_A0A1/`](results_dataaug/nested_A0A1/), local `work/aug/nested_A0A1/`)

**Controles:**
- las predicciones finales de A0 y A1 reproducen exactamente `results/A0` y `results/A1`
  (diferencia máxima 0.0);
- el α final coincide con `splits.json` en los 37 folds;
- el caché canónico no cambió.

**Selección:**
- con δ = 0 (primario), A1 en **37/37** folds;
- con δ = 0.01 (sensibilidad), A1 en 6/37.

La ventaja interna de A1 (MAE por vaca) estuvo entre −0.0112 y −0.0008 en todos los folds.

| Procedimiento | MAE fotos | MAE vaca | RMSE | R² fotos | R² vacas | ΔMAE fotos vs A0 [IC 95 %] | ΔMAE vaca vs A0 [IC 95 %] |
|---|---|---|---|---|---|---|---|
| A0 canónico | 0.583 | 0.545 | 0.682 | 0.549 | 0.519 | — | — |
| Anidado δ = 0 (= A1 siempre) | 0.576 | 0.539 | 0.678 | 0.555 | 0.517 | −0.007 [−0.021, +0.007] | −0.007 [−0.025, +0.011] |
| Anidado δ = 0.01 | 0.589 | 0.553 | 0.689 | 0.540 | 0.503 | +0.006 [+0.001, +0.013] | +0.008 [+0.001, +0.016] |

**Lectura:**

- **δ = 0.** Con la regla primaria el procedimiento siempre elige A1, así que su
  resultado es idéntico al de "A1 siempre": −0.007 BCS, con un IC que incluye 0. La
  selección interna es consistente con lo visto antes, pero no agrega evidencia de que
  la mejora generalice.
- **δ = 0.01 empeora por un artefacto del LOCO.**
  - Las 6 vacas en que eligió A1 son vacas donde A1 empeora (0/6 mejoran).
  - La causa es estructural: la ventaja interna de A1 tiene una correlación de **−0.58**
    con su efecto en la vaca excluida. Sacar del conjunto interno una vaca que A1
    perjudica aumenta la ventaja media de A1 en las 36 restantes.
  - Las reglas con umbral elegirán justo en esos folds. Es una propiedad del LOCO con
    efectos pequeños, no una evidencia en contra del flip.
- **Lo que esta etapa evalúa.** Evalúa el procedimiento de selección; no confirma de
  forma independiente una mejora. Además, como A1 se eligió como candidato después de
  ver las etapas A y B, la validación anidada no elimina ese sesgo.

**Recomendación.** Mantener **A0** como modelo principal. Ninguna etapa (A, B, C)
muestra una mejora consistente y relevante: el efecto del flip (≈ −0.007 BCS) es
estable en dirección, pero un orden de magnitud menor que la incertidumbre por animal
y menor que el umbral de ruido de ±0.02. A1 puede reportarse como hallazgo
exploratorio y volver a evaluarse con vacas nuevas.

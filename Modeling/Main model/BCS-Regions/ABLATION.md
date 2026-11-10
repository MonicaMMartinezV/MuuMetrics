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

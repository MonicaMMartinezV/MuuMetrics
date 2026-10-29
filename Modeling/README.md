# Modeling: modelo de BCS por regiones

Esta rama (`model/main`) contiene **solo el mejor modelo actual** para estimar la
condición corporal (BCS) a partir de las fotos dorsales del robot de ordeño (VMS 3).
Úsala como punto de partida para probar ideas en ramas propias (`model/DEL`,
`model/augmentation`, ...).

El código está en [`Main model/BCS-Regions/`](Main%20model/BCS-Regions/).

## 1. Cómo funciona

![Diagrama del modelo](Main%20model/BCS-Regions/figures/model_bcs.png)

1. **Segmentador de regiones**: DINOv2-B ajustado + decodificador FPN. Encuentra 6
   regiones traslapadas: lomo, cadera, grupa, base de cola, muslo izquierdo y muslo
   derecho.
2. **Descriptor**: DINOv2-L congelado, sin entrenar. Describe la foto y se
   promedia esa descripción dentro de cada región.
3. **Regresión ridge**: convierte los 6 promedios en un BCS.

Solo el ridge se entrena con nuestros datos de BCS. Por eso el modelo se entrena en
segundos y funciona con pocas vacas.

### Resultados de referencia (lo que hay que superar)

Son las 369 fotos y 37 vacas de `Dataset_Fr_BCS`. El modelo siempre se evalúa con
vacas que no vio en el entrenamiento.

| Validación | MAE | RMSE | R² fotos | R² vacas | Error ≤ 0.5 | MAE por vaca (IC 95 %) |
|---|---|---|---|---|---|---|
| Una vaca fuera (LOCO) | **0.583** | 0.682 | 0.55 | 0.52 | 56 % | 0.545 [0.465, 0.633] |
| 5 folds por vaca × 10 semillas | **0.629** | 0.729 | 0.48 | 0.46 | 50 % | 0.571 [0.482, 0.665] |
| Predecir siempre la media (LOCO) | 0.954 | | | | | |

## 2. Requisitos

- Docker con [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Para comprobarlo: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- GPU NVIDIA con unos 6 GB o más. Se probó con una RTX 2080 de 8 GB. En CPU
  funciona, pero muy lento.
- Internet la primera vez: se descargan los pesos de DINOv2-L (~1.2 GB) y quedan en
  caché en `work/.hf`.

## 3. Datos: solo `Dataset/Dataset_Fr_BCS`

**Regla del equipo: todos los modelos usan `Dataset/Dataset_Fr_BCS`.** Contiene las
vacas con ID confirmado a mano. No uses `Dataset/Images`, los CSV de DEL viejos ni
otras carpetas para entrenar o evaluar.

```
Dataset/                          <- no se sube al repo (.gitignore)
└── Dataset_Fr_BCS/
    ├── 1204_4.5/                 <- <vacaId>_<BCS>
    │   ├── ByID60__2025-06-14-11-35-07_cam4_cap2.jpg
    │   └── ...
    ├── 8758_MAYO4.25_JUNIO4_JULIO3.5/   <- BCS por mes (se toma del mes de la foto)
    └── 1552_Vaca1es2.5_Vaca2es4.75/     <- dos vacas mezcladas: se ignora
```

- La vaca y el BCS salen del nombre de la carpeta, así que no hace falta ningún CSV.
- Las fotos son los recortes dorsales del robot. El segmentador las acomoda solo.
- Hoy son 369 fotos válidas de 37 vacas, con BCS entre 2.0 y 5.0.

## 4. Pesos del segmentador

`final_all40.pt` pesa 350 MB, más de lo que GitHub acepta en el repo. Descárgalo
del Release del repositorio:

```bash
cd "Modeling/Main model/BCS-Regions"
mkdir -p weights
curl -L -o weights/final_all40.pt \
  https://github.com/MonicaMMartinezV/MuuMetrics/releases/download/bcs-regions-v1/final_all40.pt
```

## 5. Configuración (una sola vez)

Todo se corre desde `Modeling/Main model/BCS-Regions/`:

```bash
cd "Modeling/Main model/BCS-Regions"

# Archivo .env: tu usuario (para que los archivos generados no queden como root)
# y las rutas, si no usas las de por defecto
printf "UID=%s\nGID=%s\n" $(id -u) $(id -g) > .env
# echo "DATA_DIR=/ruta/a/Dataset" >> .env

docker compose build
```

| Variable | Por defecto | Qué es |
|---|---|---|
| `DATA_DIR` | `../../../Dataset` (el `Dataset/` de la raíz del repo) | Carpeta que contiene `Dataset_Fr_BCS/`; se monta en solo lectura en `/data` |
| `WEIGHTS_DIR` | `./weights` | Carpeta con `final_all40.pt` (`/weights`) |
| `WORK_DIR` | `./work` | Caché de máscaras y descriptores, resultados y caché de Hugging Face (`/work`) |
| `ORDENO_DIR` | `../../../GeneralidadesOrdeno` | Exportes del robot para calcular el DEL (`/ordeno`, solo rama `model/del`) |

El código se monta en el contenedor (`.:/app`). Si cambias un `.py`, no necesitas
volver a correr `docker compose build`; solo si cambias el `Dockerfile` o
`requirements.txt`.

## 6. Comandos

```bash
# Todo de una vez: masks + features + evaluate + train (~5 min la primera vez)
docker compose run --rm bcs all

# O paso por paso
docker compose run --rm bcs masks        # 6 máscaras por foto        -> work/masks/
docker compose run --rm bcs features     # descriptores por región    -> work/features.npz
docker compose run --rm bcs evaluate     # validación por vaca        -> work/metrics.csv, preds_*.csv
docker compose run --rm bcs train        # ridge final (todas las fotos) -> work/ridge_model.npz

# Estimar el BCS de fotos nuevas (una imagen o una carpeta)
docker compose run --rm bcs predict --input /data/Dataset_Fr_BCS/6003_2
docker compose run --rm bcs predict --input /data/fotos_nuevas --output /work/mis_preds.csv

# Ayuda con todas las opciones
docker compose run --rm bcs --help
```

| Opción | Paso | Qué hace |
|---|---|---|
| `--overwrite` | `masks` | Recalcula las máscaras aunque ya existan |
| `--seeds N` | `evaluate` | Repeticiones del esquema de 5 folds (10 por defecto) |
| `--input` | `predict` | Imagen o carpeta; las rutas son las del contenedor (`/data/...`) |
| `--model`, `--output` | `predict` | Otro `ridge_model.npz` o CSV de salida |

Las máscaras y los descriptores se guardan en caché. Si cambias algo que los
afecta, como el segmentador o el backbone, borra `work/masks/` o
`work/features.npz`, o usa `--overwrite`.

### Sin docker compose

```bash
docker build -t muumetrics-bcs-regions .
docker run --rm --gpus all --user $(id -u):$(id -g) \
  -v "$PWD":/app \
  -v /ruta/a/Dataset:/data:ro \
  -v "$PWD/weights":/weights:ro \
  -v "$PWD/work":/work \
  muumetrics-bcs-regions all
```

## 7. Archivos

| Archivo | Qué contiene |
|---|---|
| `bcs_regions.py` | Los 5 pasos (`masks`, `features`, `evaluate`, `train`, `predict`) |
| `segmenter.py` | Arquitectura del segmentador y cómo se aplica a una foto |
| `data.py` | Lectura de `Dataset_Fr_BCS` (vaca y BCS desde el nombre de la carpeta) |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Entorno con versiones fijas (torch 2.9.1 + CUDA 12.6) |
| `figures/model_bcs.*` | Diagrama del modelo (TikZ, PDF y PNG) |

## 8. Cómo proponer una mejora

1. Crea tu rama desde `model/main` con el formato `model/<idea>`:
   ```bash
   git fetch origin
   git switch -c model/augmentation origin/model/main
   ```
2. Haz tu cambio. Los lugares típicos son:
   - **Nuevas entradas** (p. ej. DEL): agrega columnas a `X` en `cmd_features`.
   - **Otra cabeza de regresión**: cambia `fit_predict`.
   - **Otro backbone o aumentos de datos**: cambia `describe` y `load_backbone`.
3. Evalúa siempre con `docker compose run --rm bcs evaluate`. Así cualquier
   cambio se compara con la misma validación por vaca.
4. En tu PR, reporta la tabla de `work/metrics.csv` contra los resultados de
   referencia de arriba.

**Reglas para que los resultados sean comparables**

- **Valida siempre por vaca.** Con una división por foto, la misma vaca queda en
  train y en test y el resultado se infla: llegamos a ver R² 0.88 por foto contra
  −0.80 por vaca.
- **Elige los hiperparámetros dentro del train de cada fold**, como hace
  `fit_predict` con alpha. Si los eliges viendo el resultado de la validación, la
  mejora es ficticia.
- **Usa solo `Dataset/Dataset_Fr_BCS`.**
- **Con 37 vacas, diferencias de ±0.02 en MAE son ruido.** Fíjate en el IC 95 %.
- **No subas datos, pesos ni `work/`.** Ya están en `.gitignore`.

## 9. Frente DEL (rama `model/del`)

Esta rama agrega el DEL (días en leche) de cada foto de `Dataset_Fr_BCS`. El DEL
se calcula a partir de los exportes diarios del robot en `GeneralidadesOrdeno/`.
Esa carpeta no se sube al repo: ponla junto a `Dataset/`, en la raíz, o indica su
ruta con `ORDENO_DIR` en el `.env`.

```
MuuMetrics/
├── Dataset/Dataset_Fr_BCS/
└── GeneralidadesOrdeno/          <- "Generalidades Ordeño_AAAAMMDD 000001.csv"
```

| Archivo | Qué contiene |
|---|---|
| `BCS-Regions/del/build_del.py` | Calcula el DEL de cada foto: vaca de la carpeta, fecha del nombre del archivo y "Días en ordeño" del archivo de Generalidades más cercano (máximo 7 días). Escribe `work/del.csv` |
| `BCS-Regions/del/del_data.py` | `load_del(names)`: DEL y lactación alineados con `features.npz` (NaN si falta; solo 1 de las 369 fotos no tiene DEL) |
| `BCS-Regions/del/del_baseline.py` | Experimento de partida: regiones solas, + DEL lineal y + spline del DEL |

```bash
docker compose run --rm bcs all                                       # si aún no tienes work/features.npz
docker compose run --rm --entrypoint python bcs del/build_del.py      # -> work/del.csv
docker compose run --rm --entrypoint python bcs del/del_baseline.py   # -> work/del_baseline.csv
```

En `del_baseline.py`, el DEL se estandariza y se multiplica por un peso w, porque
si no el ridge lo ignora frente a las 6150 columnas de la imagen. El peso y el
alpha se eligen dentro del train de cada fold.

**Resultado de partida (369 fotos, 37 vacas):** el DEL no mejora el modelo.

| Validación | Base | + DEL lineal | + DEL spline |
|---|---|---|---|
| Una vaca fuera (MAE) | **0.583** | 0.616 | 0.634 |
| 5 folds × 10 (MAE) | **0.629** | 0.630 | 0.632 |

Ninguna diferencia contra la base es significativa; el IC 95 % por bootstrap de
vacas incluye el 0. Con FiLM sobre ConvNeXt (rama `Adri`) tampoco mejoró: MAE 0.81
contra 0.82.

**Por qué cuesta:** 36 de las 37 vacas tienen un solo BCS para todas sus fotos, así
que el DEL solo puede aprender diferencias entre vacas, y la imagen ya captura casi
todo eso. La relación tiene forma de U: la vaca pierde condición después del parto,
toca fondo alrededor de 60–120 días y la recupera al final. Hay muy pocas vacas en
cada tramo; entre 120 y 200 días hay una sola.

**Ideas por probar:** interacción del DEL con cada región, distancia al punto más
bajo de la curva, la lactación como entrada, o conseguir vacas con varios BCS en
distintas etapas de la lactancia.

# Modeling: modelo de BCS por regiones

Modelo para estimar la condición corporal (BCS) a partir de las fotos dorsales del
robot de ordeño (VMS 3). Úsalo como punto de partida para probar ideas en ramas
propias (`model/DEL`, `model/augmentation`, ...).

El código está en [`Main model/BCS-Regions/`](Main%20model/BCS-Regions/). Todos los
experimentos (qué se probó, resultados y decisiones) están en
[`EXPERIMENTOS.txt`](Main%20model/BCS-Regions/EXPERIMENTOS.txt).

## 1. Cómo funciona

![Arquitectura del modelo](Main%20model/BCS-Regions/figures/arquitectura_final.png)

1. **Segmentador de regiones**: DINOv2-B ajustado + decodificador FPN, entrenado con
   40 fotos anotadas + 345 cuadros del robot pseudo-etiquetados. Encuentra 6 regiones
   traslapadas: lomo, cadera, grupa, base de cola, muslo izquierdo y muslo derecho.
2. **Descriptor**: **DINOv3-L/16 congelado** (sin entrenar), foto a **384 × 384 px**,
   salida del **bloque 21 de 24**. Los 24 × 24 tokens de parche se promedian dentro
   de cada región: 6 × 1024 + 6 indicadores de región ausente = 6150 números por foto.
3. **Regresión ridge** (alpha = 1000, elegido por validación por vaca): convierte los
   6150 números en un BCS continuo.

Solo el ridge se entrena con nuestros datos de BCS. Por eso el modelo se entrena en
segundos y funciona con pocas vacas. El ridge corre en GPU (la evaluación completa
tarda ~25 s).

### Resultados actuales (lo que hay que superar)

`Dataset_Fr_BCS_v2`: 542 fotos, 64 vacas, BCS 2.0 – 5.0. El modelo siempre se evalúa
con vacas que no vio en el entrenamiento.

| Validación | MAE | RMSE | R² fotos | R² vacas | Error ≤ 0.5 | MAE por vaca (IC 95 %) |
|---|---|---|---|---|---|---|
| Una vaca fuera (LOCO) | **0.452** | 0.565 | 0.68 | 0.63 | 71 % | 0.452 [0.390, 0.520] |
| 5 folds por vaca × 10 semillas | **0.464** | 0.570 | 0.67 | 0.63 | 70 % | 0.457 [0.394, 0.525] |
| Predecir siempre la media (LOCO) | 0.925 | | | | | |

Limitaciones conocidas:

- **Compresión hacia el promedio**: sobreestima a las vacas flacas (BCS < 3) en
  ~+0.57 y subestima a las gordas (≥ 4.5) en ~−0.34.
- **Pelaje**: a igual BCS, las vacas negras salen ~0.25 más bajas (EXPERIMENTOS [21]).
- **Lo que más lo mejora es tener más vacas**: pasar de 37 a 64 vacas bajó el MAE
  de 0.58 a 0.51; la curva de aprendizaje estima ~0.43 con 100 vacas (EXPERIMENTOS
  [16], [22]).

### Historial del modelo

| Versión | Cambio | LOCO MAE | R² |
|---|---|---|---|
| v1 (37 vacas) | DINOv2-L 378 px, último bloque | 0.583 | 0.55 |
| + segmentador con pseudo-etiquetas | [15] | 0.581 | 0.55 |
| 64 vacas (`Dataset_Fr_BCS_v2`) | [16] | 0.513 | 0.60 |
| DINOv2-L bloque 18 | [18] | 0.485 | 0.65 |
| DINOv2-L 224 px, bloque 17 | [19], [20] | 0.464 | 0.66 |
| **DINOv3-L 384 px, bloque 21 (actual)** | [24], [25] | **0.452** | **0.68** |

Lo que se probó y **no** ayudó (detalles en EXPERIMENTOS): datos externos (UNICT,
Zenodo, incluso solo Holstein), cabezas no lineales (MLP, kernel ridge), calibración,
pesos por vaca, profundidad monocular, forma de las máscaras, SigLIP 2, promediar
varias fotos de una visita, promediar con la foto en espejo, cambiar la resolución
(320–512 px, [26]) y transfer learning de BEiT (`Main model/BCS-TransferLearning/`).

## 2. Requisitos

- Docker con [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Para comprobarlo: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.
- GPU NVIDIA con unos 6 GB o más. Se probó con una RTX 2080 de 8 GB. En CPU
  funciona, pero muy lento.
- Internet la primera vez: se descargan los pesos de DINOv3-L (~1.2 GB) y quedan en
  caché en `work/.hf`.

## 3. Datos: `Dataset/Dataset_Fr_BCS_v2`

**Regla del equipo: todos los modelos usan `Dataset/Dataset_Fr_BCS_v2`.** Contiene las
vacas con ID confirmado a mano. No uses `Dataset/Images`, los CSV de DEL viejos ni
otras carpetas para entrenar o evaluar.

```
Dataset/                          <- no se sube al repo (.gitignore)
└── Dataset_Fr_BCS_v2/
    ├── 1204_4.5/                 <- <vacaId>_<BCS>
    │   ├── ByID60__2025-06-14-11-35-07_cam4_cap2.jpg
    │   └── ...
    ├── 8758_4.25/  8758_4/  8758_3.5/   <- misma vaca con BCS distinto por fecha:
    └── 9008_4.5/   9008_2.25/           <- una carpeta por calificación
```

- La vaca y el BCS salen del nombre de la carpeta, así que no hace falta ningún CSV.
  Las carpetas de la misma vaca se agrupan como **una sola vaca** en la validación.
- La 9008 bajó de 4.5 a 2.25 en ~3 semanas porque enfermó; sus etiquetas son correctas.
- Las fotos son los recortes dorsales del robot. El segmentador las acomoda solo.
- 542 fotos válidas de 64 vacas (67 carpetas), con BCS entre 2.0 y 5.0.
- La versión anterior, `Dataset_Fr_BCS` (37 vacas, 369 fotos), se puede usar con
  `--data_dir /data/Dataset_Fr_BCS`.

## 4. Pesos del segmentador

`final_all40.pt` pesa 350 MB, más de lo que GitHub acepta en el repo. Descárgalo
del Release del repositorio:

```bash
cd "Modeling/Main model/BCS-Regions"
mkdir -p weights
curl -L -o weights/final_all40.pt \
  https://github.com/MonicaMMartinezV/MuuMetrics/releases/download/bcs-regions-v2/final_all40.pt
```

> El Release `bcs-regions-v1` tiene el segmentador anterior (sin pseudo-etiquetas).
> Funciona, pero las métricas de arriba son con el de `bcs-regions-v2`.

El ridge final (`work/ridge_model.npz`, 48 KB) no se sube: se regenera con
`docker compose run --rm bcs all`.

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
| `DATA_DIR` | `../../../Dataset` (el `Dataset/` de la raíz del repo) | Carpeta que contiene `Dataset_Fr_BCS_v2/`; se monta en solo lectura en `/data` |
| `WEIGHTS_DIR` | `./weights` | Carpeta con `final_all40.pt` (`/weights`) |
| `WORK_DIR` | `./work` | Caché de máscaras y descriptores, resultados y caché de Hugging Face (`/work`) |

El código se monta en el contenedor (`.:/app`). Si cambias un `.py`, no necesitas
volver a correr `docker compose build`; solo si cambias el `Dockerfile` o
`requirements.txt`.

## 6. Comandos

```bash
# Todo de una vez: masks + features + evaluate + train (~10 min la primera vez)
docker compose run --rm bcs all

# O paso por paso
docker compose run --rm bcs masks        # 6 máscaras por foto           -> work/masks/
docker compose run --rm bcs features     # descriptores DINOv3 (~2 min)  -> work/features.npz
docker compose run --rm bcs evaluate     # validación por vaca (~25 s)   -> work/metrics.csv, preds_*.csv
docker compose run --rm bcs train        # ridge final (todas las fotos) -> work/ridge_model.npz

# Estimar el BCS de fotos nuevas (una imagen o una carpeta)
docker compose run --rm bcs predict --input /data/Dataset_Fr_BCS_v2/6003_2
docker compose run --rm bcs predict --input /data/fotos_nuevas --output /work/mis_preds.csv

# Ayuda con todas las opciones
docker compose run --rm bcs --help
```

| Opción | Paso | Qué hace |
|---|---|---|
| `--data_dir` | todos | Carpeta del dataset (por defecto `/data/Dataset_Fr_BCS_v2`) |
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
| `bcs_regions.py` | Los 5 pasos (`masks`, `features`, `evaluate`, `train`, `predict`); backbone, resolución y bloque en `BACKBONE`, `SIZE`, `LAYER` |
| `segmenter.py` | Arquitectura del segmentador y cómo se aplica a una foto |
| `data.py` | Lectura del dataset (vaca y BCS desde el nombre de la carpeta) |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | Entorno con versiones fijas (torch 2.9.1 + CUDA 12.6) |
| `EXPERIMENTOS.txt` | Registro de los experimentos [0]–[25]; cada uno indica sus archivos |
| `exp_*.py`, `prep_*.py` | Scripts de los experimentos (ver EXPERIMENTOS.txt) |
| `train_segmenter.py`, `Dockerfile.seg`, `run_seg.sh` | Entrenamiento del segmentador |
| `figures/arquitectura_final.*` | Diagrama del modelo actual (PlantUML y PNG) |
| `figures/model_bcs.*`, `proceso_completo.*`, `modelo_entrenamiento.*` | Diagramas de la versión v1 |

`Main model/BCS-TransferLearning/` contiene el modelo alternativo (BEiT-B ajustado con
las máscaras); con 37 vacas quedó muy por debajo (R² 0.03) y no se usa.

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
4. En tu PR, reporta la tabla de `work/metrics.csv` contra los resultados actuales
   de arriba, y agrega una entrada a `EXPERIMENTOS.txt`.

**Reglas para que los resultados sean comparables**

- **Valida siempre por vaca** (no por carpeta ni por foto). Con una división por
  foto, la misma vaca queda en train y en test y el resultado se infla: llegamos a
  ver R² 0.88 por foto contra −0.80 por vaca.
- **Elige los hiperparámetros dentro del train de cada fold**, como hace
  `fit_predict` con alpha. Si los eliges viendo el resultado de la validación, la
  mejora es ficticia.
- **Usa solo `Dataset/Dataset_Fr_BCS_v2`.**
- **Con 64 vacas, diferencias de ±0.02 en MAE son ruido.** Fíjate en el IC 95 % de
  la diferencia por vaca (bootstrap pareado), no solo en el MAE.
- **No subas datos, pesos ni `work/`.** Ya están en `.gitignore`.

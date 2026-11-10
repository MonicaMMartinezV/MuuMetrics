# Transfer learning: BEiT-B + máscaras de región (`model/transfer-learning`)

Ajusta **solo los últimos bloques** de un BEiT-B/16 para estimar el BCS a partir de la
foto dorsal + las 6 máscaras de `Dataset/Dataset_Fr_BCS_masks`.

La arquitectura es la del modelo de Winkler et al. 2024 (`BeitForImageClassification`,
85.8 M parámetros, cabeza de 5 clases): es exactamente `microsoft/beit-base-patch16-224`.
Sus pesos no están publicados, así que hay tres formas de inicializar el backbone:

| `--backbone` | Qué es |
|---|---|
| `imagenet` (por defecto) | `microsoft/beit-base-patch16-224`, el mismo punto de partida de Winkler |
| `/work/pretrain/<nombre>` | backbone entrenado **desde cero** (o adaptado) con `pretrain` (SimMIM, sin etiquetas) |
| `/ruta/al/checkpoint.ckpt` | el checkpoint de Winkler si lo conseguimos (state_dict con prefijo `vit.beit.`; el classifier se ignora) |

## 1. Cómo funciona

```
foto (224x224) ──► BEiT bloques 0..11-K (congelados) ──► + embedding de región ──► bloques 12-K..11 (se ajustan)
                                                         (Linear 6→768, init 0)          │
máscaras [6, 224, 224] ──► fracción de cada parche ────────────┘                        ▼
                         en cada región [196, 6]  ─────────► promedio de tokens por región (6x768) + global (768)
                                                                                         ▼
                                                                          dropout ─► Linear ─► BCS
```

- **Entrada**: la foto y el `.npz` del mismo nombre (`masks` uint8 [6, h/2, w/2]:
  loin, hips, rump, tailhead, thigh_L, thigh_R). Ambos se aplastan a cuadrado, igual que
  BCS-Regions.
- **Las máscaras** entran dos veces: como embedding sumado a los tokens antes del primer
  bloque entrenable (inicializado en cero, así al empezar el modelo es idéntico al
  preentrenado) y para promediar los tokens finales por región.
- **Regularización** (ajustar todo el modelo dio MAE 0.66):
  - solo K bloques entrenables (K ∈ `--grid`, por defecto {2, 4});
  - LP-FT: las primeras `--lp_epochs` épocas solo entrena la cabeza;
  - LLRD 0.75, drop-path 0.1, dropout 0.3, weight decay 0.05, lr del backbone 3e-5;
  - aumentos aplicados igual a foto y máscaras: volteo (intercambia muslos), rotación
    ±10°, escala 0.85–1.1, traslación, brillo/contraste/saturación;
  - **corte temprano elegido por validación interna por vaca**.
- **Hiperparámetros dentro del train de cada fold**: GroupKFold(4) por vaca sobre el
  train elige K y la época de corte (MAE por foto, como `select_alpha`). Después se
  reentrena con todo el train del fold y se predice la vaca o vacas de test (TTA con
  volteo).
- **Métricas**: se importan `data.load_records` y `bcs_regions.metrics` de
  `../BCS-Regions`, así que `metrics.csv` es directamente comparable.

## 2. Datos

```
Dataset/
├── Dataset_Fr_BCS/          <- fotos (vaca y BCS desde el nombre de la carpeta)
└── Dataset_Fr_BCS_masks/    <- mismos nombres en .npz
```

## 3. Comandos

Desde `Modeling/Main model/BCS-TransferLearning/`:

```bash
printf "UID=%s\nGID=%s\n" $(id -u) $(id -g) > .env     # una vez
docker compose build

# Evaluación (backbone ImageNet): LOCO + 5 folds x 10 semillas -> work/metrics.csv
docker compose run --rm bcs evaluate

# Entrenar BEiT DESDE CERO con fotos sin etiqueta (servidor) y evaluarlo
echo "CORPUS_DIR=/ruta/a/fotos_sin_etiqueta" >> .env
docker compose run --rm bcs pretrain --init scratch --name beit_scratch \
    --corpus /corpus/ByID /corpus/robot_visits /corpus/opencowid
docker compose run --rm bcs evaluate --backbone /work/pretrain/beit_scratch

# Variante: adaptar los pesos de ImageNet a nuestras fotos (SimMIM continuado)
docker compose run --rm bcs pretrain --init imagenet --name beit_dapt --pt_epochs 100 --lr 5e-5 \
    --corpus /corpus/ByID /corpus/robot_visits /corpus/opencowid
docker compose run --rm bcs evaluate --backbone /work/pretrain/beit_dapt

# Modelo final y predicción (npz + su foto)
docker compose run --rm bcs train
docker compose run --rm bcs predict --input /data/Dataset_Fr_BCS_masks/6003_2

docker compose run --rm bcs --help
```

- `pretrain` **excluye** cualquier foto de `Dataset_Fr_BCS` y cualquier carpeta con el
  ID de una de sus 37 vacas, para que el backbone no vea a las vacas de test. Se puede
  interrumpir: continúa desde `work/pretrain/<nombre>/last.pt`.
- `evaluate` guarda cada fold en `work/runs/<backbone>_<hash>/`. Si se interrumpe,
  continúa donde iba. `selections.csv` registra el K y la época elegidos en cada fold.
- Pruebas rápidas: `--schemes rep5 --seeds 1`.

| Opción | Por defecto | Qué hace |
|---|---|---|
| `--grid` | `2,4` | Bloques finales a ajustar (candidatos) |
| `--epochs`, `--eval_every` | 30, 3 | Épocas máximas y cada cuánto se mide la validación interna |
| `--lr_bb`, `--lr_head`, `--llrd` | 3e-5, 1e-3, 0.75 | lr del backbone, de la cabeza y decaimiento por capa |
| `--schemes`, `--seeds` | `loco,rep5`, 10 | Esquemas de validación |
| `--pt_epochs`, `--pt_batch`, `--accum`, `--lr` | 300, 32, 4, 2e-4 | SimMIM |

## 4. Tiempos (RTX 2080, 8 GB)

- `evaluate` completo: ~783 entrenamientos de ≤30 épocas, varias horas.
- `pretrain` desde cero: ~100 img/s; con ~72 k imágenes, ~12 min por época. 300
  épocas no caben en la laptop: usar el servidor.

## 5. Archivos

| Archivo | Qué contiene |
|---|---|
| `transfer.py` | Pasos `pretrain`, `evaluate`, `train`, `predict` |
| `model.py` | Carga del backbone (ImageNet / desde cero / checkpoint de Winkler), `BCSBeit`, `SimMIM` |
| `Dockerfile`, `docker-compose.yml`, `requirements.txt` | torch 2.9.1 + CUDA 12.6, transformers 4.57.1 |

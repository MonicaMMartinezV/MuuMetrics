# Correr finalModel.py (ConvNeXt) con Docker + GPU

## Prerrequisitos en el host

- Docker Engine
- Driver de NVIDIA instalado
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  configurado (`sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`)
- Verifica que funciona: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

## Estructura del dataset esperada

```
dataset/
  train/
    2.5/   imagen1.jpg  imagen2.jpg ...
    3.0/   ...
  val/
    2.5/   ...
  test/
    2.5/   ...
```

Cada subcarpeta se nombra con el score BCS (etiqueta numérica). Si tu
dataset crudo no está dividido así, primero corre
`Modeling/Support Code/split_dataset.py` para generarlo (80/10/10).

## Opción A: docker compose (recomendado)

```bash
cd "Modeling/Main model/Final model"
DATASET_DIR=/ruta/a/tu/dataset OUTPUT_DIR=/ruta/a/tus/outputs \
  docker compose up --build
```

## Opción B: docker build/run manual

```bash
cd "Modeling/Main model/Final model"

docker build -t muumetrics-convnext .

docker run --rm --gpus all \
  -v /ruta/a/tu/dataset:/data \
  -v /ruta/a/tus/outputs:/outputs \
  muumetrics-convnext \
  --data_dir /data --output_dir /outputs \
  --model convnext_base --img_size 384 \
  --batch_size 24 --epochs 40
```

Cualquier flag de `finalModel.py` (`--use_mixup`, `--use_cutmix`,
`--use_sampler`, `--freeze_epochs`, `--scheduler`, etc.) se puede pasar
al final del `docker run` o dentro de `command:` en el compose.

## Resultados

Al terminar, en tu carpeta de `outputs` local (montada en `/outputs`)
quedan:

- `best_checkpoint_by_val_mae.pth`
- `history.json`
- `preds_val.npz`, `preds_test.npz`
- `final_model.pth`

## Solo CPU (sin GPU)

Si necesitas correrlo sin GPU, cambia en el `Dockerfile` la imagen base
por `python:3.11-slim`, instala `torch`/`torchvision` sin
`--index-url` (build CPU-only) y quita el bloque `deploy.resources` del
`docker-compose.yml` (o usa `docker run` sin `--gpus all`). El
entrenamiento será considerablemente más lento.

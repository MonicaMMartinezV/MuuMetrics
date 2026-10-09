# =============================================================
# Dockerfile: entrenamiento/inferencia del modelo ConvNeXt (BCS)
#
# Imagen con soporte GPU (CUDA) para correr finalModel.py.
# Requiere en el host: NVIDIA driver + nvidia-container-toolkit,
# y correr el contenedor con `--gpus all` (o runtime: nvidia en
# docker-compose).
# =============================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema (usamos el python3 default de la imagen,
# 3.10 en esta base, para que pip instale y el ENTRYPOINT ejecute con
# el mismo intérprete — mezclar 3.10/3.11 deja los paquetes instalados
# donde el runtime no los ve)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        git \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch/torchvision se instalan aparte porque necesitan el índice
# de wheels de PyTorch con soporte CUDA. torch==2.9.1 / torchvision==0.24.1
# (versiones pineadas en requirements.txt del equipo) solo están
# publicadas para cu126/cu128/cu129, no para cu121/cu124 — de ahí el
# índice cu126 usado abajo. Si tu GPU/driver requiere otra versión de
# CUDA, ajusta el índice según https://pytorch.org/get-started/locally/
RUN pip install --upgrade pip && \
    pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY finalModel.py .

# Punto de montaje esperado para el dataset (train/val/test) y
# para los checkpoints/resultados generados
RUN mkdir -p /data /outputs

ENTRYPOINT ["python", "finalModel.py"]
CMD ["--data_dir", "/data", "--output_dir", "/outputs"]

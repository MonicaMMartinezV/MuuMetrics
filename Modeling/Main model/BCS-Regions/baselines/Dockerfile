# =============================================================
# Dockerfile de los baselines: mismo entorno que el modelo (torch 2.9.1 +
# CUDA 12.6, timm 1.0.22, requirements.txt) más Ultralytics para
# YOLOv8-cls. El modelo (Dockerfile de la carpeta de arriba) no cambia.
# Contexto de construcción: Modeling/Main model/BCS-Regions/
# =============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/work/.hf \
    YOLO_CONFIG_DIR=/work/.yolo \
    MPLCONFIGDIR=/work/.mpl

# libgl1 y libglib2.0-0: los pide opencv (dependencia de Ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libgl1 libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt ultralytics==8.3.150

RUN mkdir -p /data /work && chmod 777 /work
ENTRYPOINT ["python"]

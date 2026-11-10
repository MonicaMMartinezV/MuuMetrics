# =============================================================
# Dockerfile: transfer learning BEiT-B + máscaras (transfer.py)
#
# Requiere en el host: driver NVIDIA + nvidia-container-toolkit.
# Usa data.py y bcs_regions.py de ../BCS-Regions (montado en /regions
# por docker-compose.yml) para leer los datos y calcular las métricas
# exactamente igual que el modelo de referencia.
# =============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/work/.hf \
    PYTHONPATH=/regions

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py ./

RUN mkdir -p /data /corpus /regions /work && chmod 777 /work
ENTRYPOINT ["python", "transfer.py"]
CMD ["--help"]

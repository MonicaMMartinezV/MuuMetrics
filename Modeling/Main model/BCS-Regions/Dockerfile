# =============================================================
# Dockerfile: modelo de BCS por regiones (bcs_regions.py)
#
# Requiere en el host: driver NVIDIA + nvidia-container-toolkit y
# correr con GPU (`--gpus all` o el docker-compose.yml de esta carpeta).
# =============================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/work/.hf

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# torch 2.9.1 solo está publicado para cu126/cu128/cu129
RUN pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py ./

# /data: Dataset (solo se lee Dataset_Fr_BCS) · /weights: pesos del segmentador
# /work: caché de máscaras/features, resultados y caché de Hugging Face
RUN mkdir -p /data /weights /work && chmod 777 /work
ENTRYPOINT ["python", "bcs_regions.py"]
CMD ["--help"]

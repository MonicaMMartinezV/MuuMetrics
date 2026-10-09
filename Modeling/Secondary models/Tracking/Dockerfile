# =============================================================
# Dockerfile: Etapa 1 de re-identificación de vacas
#   - detección de vacas con YOLO (ultralytics)
#   - agrupación de detecciones en tracklets por cámara y tiempo
#
# Requiere en el host: NVIDIA driver + nvidia-container-toolkit
# y correr con `--gpus all`.
# =============================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python del sistema (3.10 en esta base) + libs que necesita OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Mismo build de torch que usamos en el entrenamiento (cu126)
RUN pip install --upgrade pip && \
    pip install torch==2.9.1 torchvision==0.24.1 \
        --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY detect_track.py assign_ids.py review_sheet.py ./

RUN mkdir -p /data/images /weights /out

ENTRYPOINT ["python", "detect_track.py"]
CMD ["--images", "/data/images", "--weights", "/weights/CowScanner.pt", "--out", "/out", "--save_crops"]

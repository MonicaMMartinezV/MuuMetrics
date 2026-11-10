#!/bin/bash
# uso: ./run_seg.sh <prep|cv|evalbase|all> [args]   (GPU, imagen muumetrics-seg)
D=$(cd "$(dirname "$0")" && pwd)
docker run --rm --gpus all --shm-size=2g -u $(id -u):$(id -g) -e HF_HOME=/work/.hf -e HOME=/work \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$D":/app -v "$D/../../../Dataset":/data:ro -v "$D/weights":/weights:ro -v "$D/work":/work \
  -v "/home/adrianmtz/Documents/MuuMetrics_Adri/Dataset/BCS_masking.sam2":/robot:ro \
  -v "/home/adrianmtz/Documents/MuuMetrics_Adri/Dataset/Images":/unlab:ro \
  -w /app --entrypoint python muumetrics-seg:latest train_segmenter.py "$@" 2>&1 | grep -vE "UserWarning|warnings.warn"

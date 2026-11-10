#!/bin/bash
# uso: ./run_depth.sh <extract|eval> [args]   (imagen muumetrics-depth = bcs-regions + transformers)
D=$(cd "$(dirname "$0")" && pwd)
GPU=$([ "$1" = "extract" ] && echo "--gpus all")
docker run --rm $GPU --shm-size=2g -u $(id -u):$(id -g) -e HF_HOME=/work/.hf -e HOME=/work -e HF_HUB_OFFLINE=1 \
  -v "$D":/app -v "$D/../../../Dataset":/data:ro -v "$D/work":/work \
  -w /app --entrypoint python muumetrics-depth:latest exp_depth.py "$@" 2>&1 | grep -vE "UserWarning|warnings.warn"

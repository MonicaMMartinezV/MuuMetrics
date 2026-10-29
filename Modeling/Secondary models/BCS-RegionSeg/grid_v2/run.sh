#!/bin/bash
# usage: run.sh <args for grid.py...>   (runs inside muumetrics-tracking with GPU)
D=$(cd "$(dirname "$0")" && pwd)
docker run --rm --gpus all -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --shm-size=2g -u $(id -u):$(id -g) -e HF_HOME=/w/.hf -e HOME=/w -v $D:/w -v $D/../bcs/.hf:/w/.hf \
  --entrypoint python muumetrics-tracking:latest /w/grid.py "$@" 2>&1 | grep -vE "Ultra|Warning|warn|UserWarning"

#!/usr/bin/env bash
# =============================================================
# run_film_cv.sh
# Entrena filmModel.py en todos los folds (split agrupado por vaca),
# con FiLM-DEL y con el baseline --no_film, y resume MAE/RMSE de test.
#
# Uso (desde esta carpeta):
#   ./run_film_cv.sh                      -> 5 folds x {film, nofilm}
#   FOLDS="0 1" ./run_film_cv.sh --epochs 20 --use_sampler
#   (cualquier flag extra se pasa a filmModel.py)
#
# Variables: IMAGES_DIR, DEL_CSV, OUTPUT_DIR, FOLDS, N_FOLDS
# =============================================================
set -euo pipefail

REPO=$(cd ../../.. && pwd)
IMAGES_DIR=${IMAGES_DIR:-$REPO/Dataset/Images}
DEL_CSV=${DEL_CSV:-$REPO/dataset_del.csv}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO/Dataset/outputs_film_cv}
N_FOLDS=${N_FOLDS:-5}
FOLDS=${FOLDS:-$(seq -s ' ' 0 $((N_FOLDS - 1)))}
mkdir -p "$OUTPUT_DIR"

for fold in $FOLDS; do
    for variant in film nofilm; do
        flag=(); [[ $variant == nofilm ]] && flag=(--no_film)
        echo "=== fold $fold / $variant ==="
        docker run --rm --gpus all --shm-size=2g -u "$(id -u):$(id -g)" \
            -e HOME=/tmp -e TORCH_HOME=/tmp/torch -e HF_HOME=/tmp/hf \
            --name "muumetrics-training-film-f${fold}-${variant}" \
            -v "$IMAGES_DIR":/data/images:ro \
            -v "$DEL_CSV":/data/dataset_del.csv:ro \
            -v "$OUTPUT_DIR":/outputs \
            --entrypoint python muumetrics-convnext:latest filmModel.py \
            --csv /data/dataset_del.csv --images_dir /data/images \
            --output_dir "/outputs/${variant}/fold${fold}" \
            --n_folds "$N_FOLDS" --fold "$fold" "${flag[@]}" "$@"
    done
done

python3 - "$OUTPUT_DIR" <<'PY'
import json, sys, glob, statistics as st
out = sys.argv[1]
for variant in ("film", "nofilm"):
    mets = [json.load(open(p)) for p in sorted(glob.glob(f"{out}/{variant}/fold*/test_metrics.json"))]
    if not mets: continue
    s = {k: (st.mean(m[k] for m in mets), st.pstdev(m[k] for m in mets)) for k in ("mae", "rmse", "r2")}
    print(f"{variant:7s} n={len(mets)}  " + "  ".join(f"{k}={m:.4f}±{sd:.4f}" for k, (m, sd) in s.items()))
PY

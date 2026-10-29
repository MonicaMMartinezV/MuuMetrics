#!/bin/bash
# Learning curve after finalize: best config with 8/15/22 training images per fold, then plot and copy to repo
cd "$(dirname "$0")"; R="/home/adrianmtz/Documents/MuuMetrics_Adri/Modeling/Secondary models/BCS-RegionSeg/grid_v2"
while kill -0 1620523 2>/dev/null || docker ps --format '{{.Command}}' | grep -q grid.py; do sleep 60; done
echo "[lc] start ($(date))"
for n in 8 15 22; do ./run.sh --enc vit_base_patch14_dinov2.lvd142m --mode finetune --enc_lr_mult 0.1 --lr 0.0003 --h 378 --w 672 --iters 2000 --bs 4 --loss bcedice --dec 128 --ntrain $n --tag lc_n$n; done
docker run --rm -u $(id -u):$(id -g) -v $PWD:/w --entrypoint python muumetrics-tracking:latest /w/plot_lc.py 2>&1 | grep -v Ultra
mkdir -p "$R/results" "$R/figs"; cp results/lc_*.json "$R/results/"; cp figs/learning_curve.png "$R/figs/"; cp grid.py plot_lc.py lc.sh viz_pick.py "$R/"
echo "[lc] ALL DONE ($(date))"

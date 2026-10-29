#!/bin/bash
# Runs after stage 2: posthoc, pick best, final model (37 -> test), production model (all 40), figures, copy to repo
cd "$(dirname "$0")"; R="/home/adrianmtz/Documents/MuuMetrics_Adri/Modeling/Secondary models/BCS-RegionSeg/grid_v2"
CPU="docker run --rm -u $(id -u):$(id -g) -v $PWD:/w --entrypoint python muumetrics-tracking:latest"
echo "[finalize] waiting for stage 2 ($(date))"
while kill -0 1534744 2>/dev/null || docker ps --format '{{.Command}}' | grep -q grid.py; do sleep 60; done
echo "[finalize] stage 2 done ($(date))"
$CPU /w/posthoc.py 2>&1 | grep -v Ultra > posthoc.txt
python3 pick_best.py; BEST=$(cat best_tag.txt); ARGS=$(cat best_args.txt)
./run.sh $ARGS --final --tag final_test
./run.sh $ARGS --all --tag final_all40
$CPU /w/viz_oof.py "$BEST" 2>&1 | grep -v Ultra
$CPU /w/viz_oof.py final_test 2>&1 | grep -v Ultra
mkdir -p "$R/results" "$R/figs" "$R/weights"
cp grid.py rle.py label.py prep.py sheet.py posthoc.py viz_oof.py pick_best.py run.sh finalize.sh labels.json summary.txt posthoc.txt best_args.txt "$R/"
cp results/*.json "$R/results/"; cp figs/*.jpg sheet*.jpg "$R/figs/"; cp results/*.pt "$R/weights/"
printf '*.pt\n' > "$R/.gitignore"
echo "[finalize] ALL DONE ($(date)) -> $R"

#!/bin/bash
# Submit one train+test job per seed. Run on Fox from the repo root:
#     bash AVST/fox/submit_seeds.sh                                    # ResNet-18 baseline, seeds 1 2 3
#     NAME=dinov2_b FEATS=AVST/data/feats/dinov2_vitb14_reg bash AVST/fox/submit_seeds.sh
# When all jobs are done:
#     python AVST/tools/summarize_results.py --label "$NAME" results/avst_${NAME}_seed*.json
set -euo pipefail

NAME=${NAME:-resnet18}
FEATS=${FEATS:-AVST/data/feats/res18_14x14}
SEEDS=${SEEDS:-"1 2 3"}

mkdir -p logs results
for SEED in $SEEDS; do
    sbatch --job-name="avst-$NAME-s$SEED" --export=ALL,SEED="$SEED",NAME="$NAME",FEATS="$FEATS" AVST/fox/run_avst.slurm
done

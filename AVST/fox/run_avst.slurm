#!/bin/bash
# Train and/or test AVST on Fox. Submit from the repo root (use submit_seeds.sh for several seeds):
#     mkdir -p logs results
#     sbatch --export=ALL,SEED=1 AVST/fox/run_avst.slurm                         # train 30 epochs + test
#     sbatch --export=ALL,MODE=test,CKPT=avst_ours AVST/fox/run_avst.slurm       # only test an existing checkpoint
# Variables (all optional):
#     SEED   random seed                              (default 1)
#     NAME   backbone/experiment label                (default resnet18)
#     FEATS  visual feature dir                       (default AVST/data/feats/res18_14x14)
#     MODE   train_test | test                        (default train_test)
#     CKPT   checkpoint name in avst_models/, no .pt  (default avst_${NAME}_seed${SEED})
#SBATCH --job-name=avst
#SBATCH --account=ec12
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

SEED=${SEED:-1}
NAME=${NAME:-resnet18}
FEATS=${FEATS:-AVST/data/feats/res18_14x14}
MODE=${MODE:-train_test}
CKPT=${CKPT:-avst_${NAME}_seed${SEED}}
PY_MODULE=${PY_MODULE:-Python/3.11.3-GCCcore-12.3.0}
VENV=${VENV:-/fp/projects01/ec12/$USER/venvs/avst}

module purge
module load "$PY_MODULE"
source "$VENV/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p results
echo "job $SLURM_JOB_ID on $(hostname)  seed=$SEED name=$NAME feats=$FEATS mode=$MODE ckpt=$CKPT"
nvidia-smi --query-gpu=name,memory.total --format=csv

COMMON=(--label_train AVST/data/json/avqa-train.json
        --label_val AVST/data/json/avqa-val.json
        --label_test AVST/data/json/avqa-test.json
        --audio_dir AVST/data/feats/vggish
        --video_res14x14_dir "$FEATS"
        --model_save_dir AVST/net_grd_avst/avst_models/
        --checkpoint "$CKPT"
        --gpu "${CUDA_VISIBLE_DEVICES:-0}"
        --seed "$SEED")

if [ "$MODE" = "train_test" ]; then
    python AVST/net_grd_avst/main_avst.py "${COMMON[@]}" --mode train \
        --batch-size 16 --epochs 30 --num_workers "${SLURM_CPUS_PER_TASK:-8}"
fi

python AVST/net_grd_avst/main_avst.py "${COMMON[@]}" --mode test --pred_out "results/${CKPT}.json"

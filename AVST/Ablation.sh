#!/bin/bash

#SBATCH --job-name=ablate
#SBATCH --account=ec12
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=ablate-%j.out

export PYTHONUNBUFFERED=1
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /fp/projects01/ec12/ec-williaan/venvs/avst/bin/activate

# repo root
DIR=/fp/projects01/ec12/ec-williaan/MusiQAl
cd $DIR

# tests the trained models with one input replaced (zeros, noise, or another video's input). no training.
# one seed per job, run all three in parallel:
#   for s in 1 2 3; do sbatch --export=ALL,SEED=$s AVST/Ablation.sh; done
# then: python AVST/tools/summarize_results.py ablate-*.out
SEED=${SEED:-1}
ABLATIONS=${ABLATIONS:-"visual_zero visual_noise visual_shuffle audio_zero audio_noise audio_shuffle both_zero both_shuffle"}

# name, features, visual_dim, temporal layers
SETUPS="avst,res18_14x14,512,0 avst_dinov2,dinov2_vitb14_reg,768,0 avst_dinov2_grd,dinov2_vitb14_reg,768,0 avst_dinov2_grd_temp,dinov2_vitb14_reg,768,2"

for setup in $SETUPS; do
    IFS=, read NAME FEATS VDIM TLAYERS <<< "$setup"
    for ABL in $ABLATIONS; do
        echo "=== $NAME seed $SEED $ABL"
        python AVST/net_grd_avst/main_avst.py --label_test AVST/data/json/avqa-test.json --mode test --ablate $ABL --video_res14x14_dir AVST/data/feats/$FEATS --visual_dim $VDIM --temporal_layers $TLAYERS --checkpoint ${NAME}_seed$SEED --model_save_dir AVST/net_grd_avst/avst_models/ --gpu ${CUDA_VISIBLE_DEVICES:-0}
    done
done

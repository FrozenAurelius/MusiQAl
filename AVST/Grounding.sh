#!/bin/bash

#SBATCH --job-name=grd
#SBATCH --account=ec12
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=grd-%j.out

export PYTHONUNBUFFERED=1
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /fp/projects01/ec12/ec-williaan/venvs/avst/bin/activate

# repo root
DIR=/fp/projects01/ec12/ec-williaan/MusiQAl
cd $DIR

# grounding stage on dinov2 feats, starting from the resnet18 grounding stage.
# saves AVST/grounding_gen/models_grounding_gen/grounding_dinov2_best.pt, then:
#   sbatch --export=ALL,FEATS=dinov2_vitb14_reg,VDIM=768,NAME=avst_dinov2_grd,GRD=AVST/grounding_gen/models_grounding_gen/grounding_dinov2_best.pt AVST/Test.sh
EPOCHS=${EPOCHS:-10}

python AVST/grounding_gen/main_grd_gen_feat.py --video_feat_dir AVST/data/feats/dinov2_vitb14_reg --visual_dim 768 --epochs $EPOCHS --checkpoint grounding_dinov2

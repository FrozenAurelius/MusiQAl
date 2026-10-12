#!/bin/bash

#SBATCH --job-name=avst
#SBATCH --account=ec12
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=avst-%j.out

module purge
module load Python/3.10.4-GCCcore-11.3.0
source /fp/projects01/ec12/ec-williaan/venvs/avst/bin/activate

# repo root (the code uses paths relative to it, e.g. ./AVST/data/...)
DIR=/fp/projects01/ec12/ec-williaan/MusiQAl
cd $DIR

# sbatch --export=ALL,SEED=2 AVST/Test.sh  to train another seed
SEED=${SEED:-1}
EPOCHS=${EPOCHS:-30}   # sbatch --export=ALL,EPOCHS=1 AVST/Test.sh  for a quick check

python AVST/net_grd_avst/main_avst.py --label_train AVST/data/json/avqa-train.json --label_val AVST/data/json/avqa-val.json --label_test AVST/data/json/avqa-test.json --batch-size 16 --epochs $EPOCHS --mode train --seed $SEED --checkpoint avst_seed$SEED --model_save_dir AVST/net_grd_avst/avst_models/ --gpu ${CUDA_VISIBLE_DEVICES:-0}

python AVST/net_grd_avst/main_avst.py --label_train AVST/data/json/avqa-train.json --label_val AVST/data/json/avqa-val.json --label_test AVST/data/json/avqa-test.json --mode test --checkpoint avst_seed$SEED --model_save_dir AVST/net_grd_avst/avst_models/ --gpu ${CUDA_VISIBLE_DEVICES:-0}

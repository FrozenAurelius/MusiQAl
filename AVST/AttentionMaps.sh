#!/bin/bash

#SBATCH --job-name=att-maps
#SBATCH --account=ec12
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --output=att-%j.out

export PYTHONUNBUFFERED=1
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /fp/projects01/ec12/ec-williaan/venvs/avst/bin/activate

# repo root
DIR=/fp/projects01/ec12/ec-williaan/MusiQAl
cd $DIR

# heatmaps of where the seed-1 models look for the sound, saved to AVST/attention_maps/<video>.png
# other videos:  sbatch --export=ALL,VIDEOS="00000155 00000034" AVST/AttentionMaps.sh
if [ -n "$VIDEOS" ]; then
    python AVST/tools/attention_maps.py --videos $VIDEOS
else
    python AVST/tools/attention_maps.py --n_videos 8
fi

#!/bin/bash

#SBATCH --job-name=feat-viz
#SBATCH --account=ec12
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=viz-%j.out

export PYTHONUNBUFFERED=1
module purge
module load Python/3.10.4-GCCcore-11.3.0
source /fp/projects01/ec12/ec-williaan/venvs/avst/bin/activate

# repo root
DIR=/fp/projects01/ec12/ec-williaan/MusiQAl
cd $DIR

# compares the saved resnet18 and dinov2 features (no GPU needed), output in AVST/feature_viz/
python AVST/tools/feature_viz.py

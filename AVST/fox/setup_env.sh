#!/bin/bash
# One-time Python environment setup on Fox. Run on a login node from the repo root:
#     bash AVST/fox/setup_env.sh
# If the module below does not exist, pick another Python 3.10/3.11 from `module avail Python`
# and use the same name in run_avst.slurm.
set -euo pipefail

PY_MODULE=${PY_MODULE:-Python/3.11.3-GCCcore-12.3.0}
VENV=${VENV:-/fp/projects01/ec12/$USER/venvs/avst}

module purge
module load "$PY_MODULE"

python -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
# CUDA wheels bundle their own CUDA runtime, so no CUDA module is needed.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas munch tensorboard scipy matplotlib

python -c "import torch, torchvision; print('torch', torch.__version__, 'torchvision', torchvision.__version__)"
echo "Environment ready: $VENV"

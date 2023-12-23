#!/usr/bin/bash
#SBATCH --job-name=CLIFF_RICH
#SBATCH --output=CLIFF_RICH.out
#SBATCH --error=CLIFF_RICH.err
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --gpus=1

python3 eval.py



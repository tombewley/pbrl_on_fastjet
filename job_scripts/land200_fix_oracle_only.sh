#!/bin/bash -login

#SBATCH --job-name=pbrl_land200_fix_oracle_only
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00

PARAMS=(
    "fastjet.land pets=3 200 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping"
)

module load CUDA
source activate env_pytorch

srun python run.py fastjet.land pets=3 200 --wandb=1 --group=land200_fix --oracle=closing_and_shaping

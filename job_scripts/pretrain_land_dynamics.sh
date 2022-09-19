#!/bin/bash -login

#SBATCH --job-name=pretrain_land_dynamics
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=12:00:00

module load CUDA
source activate env_pytorch

srun python scripts/pretrain_dynamics.py land 2

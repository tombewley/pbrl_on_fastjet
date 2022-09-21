#!/bin/bash -login

#SBATCH --job-name=rollout_final_models_chase
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=24:00:00

module load CUDA
source activate env_pytorch

srun python scripts/evaluate/rollout_final_models.py chase dist20_los_roll_alt50 --num_eps=500

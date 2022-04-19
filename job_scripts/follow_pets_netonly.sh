#!/bin/bash -login

#SBATCH --job-name=follow_pets
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=05:00:00

module load CUDA
source activate env_pytorch

srun python run.py task.follow oracle=dist_closing_uperr_v2 wandb agent.pets schedule.200_one_per_ep model.net sampler.ucb_nostd_recency

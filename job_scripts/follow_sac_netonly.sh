#!/bin/bash -login

#SBATCH --job-name=follow_sac
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=15:00:00

module load CUDA
source activate env_pytorch

srun python run.py task.follow oracle=dist_closing_uperr_v2 wandb agent.sac schedule.100k_batch model.net sampler.ucb_nostd_recency

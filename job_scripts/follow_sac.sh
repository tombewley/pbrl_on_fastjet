#!/bin/bash -login

#SBATCH --job-name=follow_sac
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=15:00:00
#SBATCH --array=0-2

PARAMS=(
    "agent.sac schedule.100k_batch model.tree_by_variance sampler.ucb_nostd_recency"
    "agent.sac schedule.100k_batch model.net sampler.ucb_nostd_recency"
    "agent.sac schedule.100k"
)

module load CUDA
source activate env_pytorch

srun python run.py task.follow oracle=dist_closing_uperr wandb ${PARAMS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash -login

#SBATCH --job-name=follow_pets
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=05:00:00
#SBATCH --array=0-3

PARAMS=(
    "agent.pets schedule.200_one_per_ep model.tree_by_variance sampler.entropy_recency"
    "agent.pets schedule.200_one_per_ep model.tree_by_variance sampler.uniform_recency"
    "agent.pets schedule.200_one_per_ep model.tree_by_variance sampler.ucb_nostd_norecency"
    "agent.pets schedule.200"
)

module load CUDA
conda init bash
conda activate env_pytorch

srun python run.py task.follow oracle=dist_closing_uperr wandb ${PARAMS[$SLURM_ARRAY_TASK_ID]}

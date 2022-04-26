#!/bin/bash -login

#SBATCH --job-name=target_hard_multi_pruning
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=5:00:00
#SBATCH --array=0-6

PARAMS=(
    "model.tree sweep.multi_pruning=0",
    "model.tree sweep.multi_pruning=1",
    "model.tree sweep.multi_pruning=2",
    "model.tree sweep.multi_pruning=3",
    "model.tree sweep.multi_pruning=4",
    "model.tree sweep.multi_pruning=5",
    "model.net"
)

module load CUDA
source activate env_pytorch

srun python run.py wandb task.target_hard oracle=dist_pose_when_close schedule.1k_freq10_notrain agent.pets_fast sampler.ucb_nostd_recency ${PARAMS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash -login

#SBATCH --job-name=pbrl_220902
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=2:00:00
#SBATCH --array=0-1

PARAMS=(
    "env.fastjet.follow agent.pets=3 oracle.fastjet.follow.dist_closing_uperr_v2 features.fastjet.default sampler.ucb_nostd_norecency schedule.200_one_per_ep wandb model.tree.fullgraph=4"
    "env.fastjet.follow agent.pets=3 oracle.fastjet.follow.dist_closing_uperr_v2 features.fastjet.default sampler.ucb_nostd_norecency schedule.200_one_per_ep wandb model.tree.fullgraph=5"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

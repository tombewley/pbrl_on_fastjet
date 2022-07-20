#!/bin/bash -login

#SBATCH --job-name=pets_planning_sweep_on_follow
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=5:00:00
#SBATCH --array=0-3

module load CUDA
source activate env_pytorch

srun python run.py misc.pets_planning_sweep_on_follow oracle.fastjet.follow.dist_closing_uperr_v2 agent.pets=${SLURM_ARRAY_TASK_ID}

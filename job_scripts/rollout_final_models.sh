#!/bin/bash -login

#SBATCH --job-name=rollout_final_models
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=12:00:00
#SBATCH --array=0-2

PARAMS=(
    "match dist_pose_when_close"
    "follow dist_closing_uperr_v2"
    "chase dist20_los_roll_alt50"
)

module load CUDA
source activate env_pytorch

srun python scripts/evaluate/rollout_final_models.py ${PARAMS[$SLURM_ARRAY_TASK_ID]} --num_eps=500

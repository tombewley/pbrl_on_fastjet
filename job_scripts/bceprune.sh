#!/bin/bash -login

#SBATCH --job-name=pbrl_bceprune
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --array=0-3

PARAMS=(
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=11"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=11"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=11"
    "fastjet.land pets=3 200 --wandb=1 --group=land200 --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=11"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

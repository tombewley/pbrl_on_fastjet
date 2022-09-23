#!/bin/bash -login

#SBATCH --job-name=pbrl_linear_models
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=4:00:00
#SBATCH --array=0-2

PARAMS=(
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=linear"
    "fastjet.chase pets=3 200 --wandb=1 --group=chase_with_alt --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=linear"
    "fastjet.land pets=3 200 --wandb=1 --group=land200_fix --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=linear"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

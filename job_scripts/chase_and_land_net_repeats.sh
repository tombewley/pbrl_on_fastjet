#!/bin/bash -login

#SBATCH --job-name=pbrl_chase_and_land_net_repeats
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --array=0-1

PARAMS=(
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=land200_fix --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

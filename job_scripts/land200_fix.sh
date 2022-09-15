#!/bin/bash -login

#SBATCH --job-name=pbrl_land200_fix
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --array=0-2

PARAMS=(
    "fastjet.land pets=3 200 --wandb=1 --group=land200_fix --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=land200_fix --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=1"
    "fastjet.land pets=3 200 --wandb=1 --group=land200_fix --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash -login

#SBATCH --job-name=pbrl_land200
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=4:00:00

source activate env_pytorch

srun python run.py fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net

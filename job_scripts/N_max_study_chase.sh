#!/bin/bash -login

#SBATCH --job-name=pbrl_N_max_study_chase
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=10:00:00
#SBATCH --array=0-7

PARAMS=(
    "fastjet.chase pets=3 50 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=tree=10"
    "fastjet.chase pets=3 100 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=tree=10"
    "fastjet.chase pets=3 400 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=10"
    "fastjet.chase pets=3 800 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=tree=10"
    "fastjet.chase pets=3 50 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=net"
    "fastjet.chase pets=3 100 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=net"
    "fastjet.chase pets=3 400 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=net"
    "fastjet.chase pets=3 800 --wandb=1 --group=N_max_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=net"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

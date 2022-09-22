#!/bin/bash -login

#SBATCH --job-name=pbrl_beta_study
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=5:00:00
#SBATCH --array=0-9

PARAMS=(
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=2"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=3"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=4"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=5"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=beta_chase=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=beta_chase=2"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=beta_chase=3"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=beta_chase=4"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=beta_chase=5"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

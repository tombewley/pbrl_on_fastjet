#!/bin/bash -login

#SBATCH --job-name=pbrl_robustness_pets
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=10:00:00
#SBATCH --array=0-8

PARAMS=(
    "fastjet.match pets=2 200 --wandb=1 --group=robustness_pets_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.match pets=4 200 --wandb=1 --group=robustness_pets_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.match pets=5 200 --wandb=1 --group=robustness_pets_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.follow pets=2 200 --wandb=1 --group=robustness_pets_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.follow pets=4 200 --wandb=1 --group=robustness_pets_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.follow pets=5 200 --wandb=1 --group=robustness_pets_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.chase pets=2 200 --wandb=1 --group=robustness_pets_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.chase pets=4 200 --wandb=1 --group=robustness_pets_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.chase pets=5 200 --wandb=1 --group=robustness_pets_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

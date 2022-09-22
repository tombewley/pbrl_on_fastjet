#!/bin/bash -login

#SBATCH --job-name=pbrl_gamma_study
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=5:00:00
#SBATCH --array=0-31

PARAMS=(
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=0"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=1"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=2"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=3"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=0"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=1"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=2"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=3"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=0"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=2"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=3"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=0"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=1"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=2"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=gamma=3"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=0"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=1"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=2"
    "fastjet.match pets=3 200 --wandb=1 --group=gamma_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=3"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=0"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=1"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=2"
    "fastjet.follow pets=3 200 --wandb=1 --group=gamma_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=3"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=0"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=2"
    "fastjet.chase pets=3 200 --wandb=1 --group=gamma_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=3"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=0"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=1"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=2"
    "fastjet.land pets=3 200 --wandb=1 --group=gamma_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net --irrationality=gamma=3"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

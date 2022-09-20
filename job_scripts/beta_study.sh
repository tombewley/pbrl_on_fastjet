#!/bin/bash -login

#SBATCH --job-name=pbrl_beta_study
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=5:00:00
#SBATCH --array=0-19

PARAMS=(
    "fastjet.match pets=3 200 --wandb=1 --group=beta_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_match=1"
    "fastjet.match pets=3 200 --wandb=1 --group=beta_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_match=2"
    "fastjet.match pets=3 200 --wandb=1 --group=beta_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_match=3"
    "fastjet.match pets=3 200 --wandb=1 --group=beta_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_match=4"
    "fastjet.match pets=3 200 --wandb=1 --group=beta_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_match=5"
    "fastjet.follow pets=3 200 --wandb=1 --group=beta_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_follow=1"
    "fastjet.follow pets=3 200 --wandb=1 --group=beta_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_follow=2"
    "fastjet.follow pets=3 200 --wandb=1 --group=beta_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_follow=3"
    "fastjet.follow pets=3 200 --wandb=1 --group=beta_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_follow=4"
    "fastjet.follow pets=3 200 --wandb=1 --group=beta_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_follow=5"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=2"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=3"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=4"
    "fastjet.chase pets=3 200 --wandb=1 --group=beta_study_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_chase=5"
    "fastjet.land pets=3 200 --wandb=1 --group=beta_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_land=1"
    "fastjet.land pets=3 200 --wandb=1 --group=beta_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_land=2"
    "fastjet.land pets=3 200 --wandb=1 --group=beta_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_land=3"
    "fastjet.land pets=3 200 --wandb=1 --group=beta_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_land=4"
    "fastjet.land pets=3 200 --wandb=1 --group=beta_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10 --irrationality=beta_land=5"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

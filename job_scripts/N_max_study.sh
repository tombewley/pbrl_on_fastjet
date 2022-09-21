#!/bin/bash -login

#SBATCH --job-name=pbrl_N_max_study
#SBATCH --nodes=1
#SBATCH --partition veryshort
#SBATCH --time=6:00:00
#SBATCH --array=0-23

PARAMS=(
    "fastjet.match pets=3 50 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=tree=10"
    "fastjet.match pets=3 100 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=tree=10"
    "fastjet.match pets=3 400 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=10"
    "fastjet.match pets=3 800 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=tree=10"
    "fastjet.follow pets=3 50 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=tree=10"
    "fastjet.follow pets=3 100 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=tree=10"
    "fastjet.follow pets=3 400 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=10"
    "fastjet.follow pets=3 800 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=tree=10"
    "fastjet.land pets=3 50 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=tree=10"
    "fastjet.land pets=3 100 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=tree=10"
    "fastjet.land pets=3 400 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=10"
    "fastjet.land pets=3 800 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=tree=10"
    "fastjet.match pets=3 50 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=net"
    "fastjet.match pets=3 100 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=net"
    "fastjet.match pets=3 400 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=net"
    "fastjet.match pets=3 800 --wandb=1 --group=N_max_study_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=net"
    "fastjet.follow pets=3 50 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=net"
    "fastjet.follow pets=3 100 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=net"
    "fastjet.follow pets=3 400 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=net"
    "fastjet.follow pets=3 800 --wandb=1 --group=N_max_study_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=net"
    "fastjet.land pets=3 50 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_50_1_1 --model=net"
    "fastjet.land pets=3 100 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_100_1_1 --model=net"
    "fastjet.land pets=3 400 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=net"
    "fastjet.land pets=3 800 --wandb=1 --group=N_max_study_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_800_1_1 --model=net"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

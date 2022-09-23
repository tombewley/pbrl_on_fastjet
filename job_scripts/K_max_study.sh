#!/bin/bash -login

#SBATCH --job-name=pbrl_K_max_study
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=24:00:00
#SBATCH --array=0-19

PARAMS=(
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=250_200_1_1 --model=tree=10"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=500_200_1_1 --model=tree=10"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=2k_200_1_1 --model=tree=10"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=4k_200_1_1 --model=tree=10"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=250_200_1_1 --model=tree=10"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=500_200_1_1 --model=tree=10"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=2k_200_1_1 --model=tree=10"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=4k_200_1_1 --model=tree=10"
    "fastjet.follow pets=3 200 --wandb=1 --group=robustness_budget_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=250_200_1_1 --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=robustness_budget_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=500_200_1_1 --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=robustness_budget_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=2k_200_1_1 --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=robustness_budget_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=4k_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=250_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=500_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=2k_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=robustness_budget_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=4k_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=250_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=500_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=2k_200_1_1 --model=net"
    "fastjet.land pets=3 200 --wandb=1 --group=robustness_budget_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=4k_200_1_1 --model=net"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

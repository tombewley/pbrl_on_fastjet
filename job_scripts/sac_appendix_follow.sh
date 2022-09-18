#!/bin/bash -login

#SBATCH --job-name=sac_appendix_follow
#SBATCH --nodes=1
#SBATCH --partition cpu
#SBATCH --time=24:00:00
#SBATCH --array=0-3

PARAMS=(
    "fastjet.follow sac 50000 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2"
    "fastjet.follow sac 50000 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_50k_250_250 --model=net"
    "fastjet.follow sac 50000 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_50k_250_250 --model=tree=1"
    "fastjet.follow sac 50000 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_50k_250_250 --model=tree=10"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

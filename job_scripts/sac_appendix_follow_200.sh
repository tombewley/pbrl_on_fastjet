#!/bin/bash -login

#SBATCH --job-name=sac_appendix_follow_200
#SBATCH --nodes=1
#SBATCH --partition cpu
#SBATCH --time=24:00:00
#SBATCH --array=0-3

PARAMS=(
    "fastjet.follow sac_smallmem 200 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.follow sac_smallmem 200 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=1"
    "fastjet.follow sac_smallmem 200 --wandb=1 --group=sac_appendix_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

#!/bin/bash -login

#SBATCH --job-name=pbrl_220902_extra
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --array=0-15

PARAMS=(
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=1"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=1"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1 --model=tree=10"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=net"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=1"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1 --model=tree=10"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=1"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-09_match --save_freq=50 --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=10"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=1"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-09_follow --save_freq=50 --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=10"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=1"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-09_chase --save_freq=50 --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=10"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=net"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=1"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-09_land --save_freq=50 --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=10"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

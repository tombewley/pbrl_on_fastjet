#!/bin/bash -login

#SBATCH --job-name=pbrl_220902
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --partition gpu
#SBATCH --time=4:00:00
#SBATCH --array=0-43

PARAMS=(
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=net"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=0"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=4"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=8"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=12"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=0"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=4"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=8"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=12"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=0"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=4"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=8"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=ucb_nostd_norecency --schedule=1k_200_1_1 --model=tree=12"
    "fastjet.land pets=3 200 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=ucb_nostd_norecency --schedule=1k_400_1_1 --model=net"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=ucb_nostd_norecency --schedule=1k_400_1_1 --model=tree=0"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=ucb_nostd_norecency --schedule=1k_400_1_1 --model=tree=4"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=ucb_nostd_norecency --schedule=1k_400_1_1 --model=tree=8"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=ucb_nostd_norecency --schedule=1k_400_1_1 --model=tree=12"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=0"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=4"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=8"
    "fastjet.match pets=3 200 --wandb=1 --group=22-09-02_match --oracle=dist_pose_when_close --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=12"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=0"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=4"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=8"
    "fastjet.follow pets=3 200 --wandb=1 --group=22-09-02_follow --oracle=dist_closing_uperr_v2 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=12"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=net"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=0"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=4"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=8"
    "fastjet.chase pets=3 200 --wandb=1 --group=22-09-02_chase --oracle=dist20_los_roll_alt50 --features=default --sampler=uniform_recency --schedule=1k_200_1_1_sched --model=tree=12"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=net"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=0"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=4"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=8"
    "fastjet.land pets=3 400 --wandb=1 --group=22-09-02_land --oracle=closing_and_shaping --features=default --sampler=uniform_recency --schedule=1k_400_1_1_sched --model=tree=12"
)

module load CUDA
source activate env_pytorch

srun python run.py ${PARAMS[$SLURM_ARRAY_TASK_ID]}

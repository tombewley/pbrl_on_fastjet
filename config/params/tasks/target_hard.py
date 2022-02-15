SKIP_FRAMES = 25
P = {"deployment": {
    "task": "target_no_reward_hard",
    "num_episodes": 100000,
    "episode_time_limit": 750 / SKIP_FRAMES,
    "skip_frames": SKIP_FRAMES
    },
    "agent": {"sac": {"gamma": 0.9}}
}
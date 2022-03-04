SKIP_FRAMES = 25

P = {
    "deployment": {
        "project_name": "fastjet",
        "task": "target_no_reward",
        "train": True,
        "agent": "sac",
        "num_episodes": 100000,
        "episode_time_limit": 500 / SKIP_FRAMES, # Num frames = episode_time_limit * skip_frames
        "skip_frames": SKIP_FRAMES
    },
    "agent": {
        "sac": {
            "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
            "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
            "input_normaliser": "obs_lims",
            "replay_capacity": 5e5,
            "batch_size": 32, 
            "lr_pi": 1e-4, 
            "lr_Q": 1e-3,
            "gamma": 0.99, 
            "alpha": 0.2, 
            "tau": 0.005, 
            "update_freq": 1 
        }
    },
    "pbrl": {
        "reward_source": "extrinsic"
    }  
}
P = {
    "agent": {
        "sac": { # Ian's settings
            "net_pi": [(None, 512), "R", (512, 512), "R", (512, 256), "R", (256, None)],
            "net_Q": [(None, 512), "R", (512, 512), "R", (512, 256), "R", (256, None)],
            "replay_capacity": 1e6,
            "batch_size": 1024,
            "lr_pi": 5.0e-5,
            "lr_Q": 5.0e-5,
            "gamma": 0.95,
            "alpha": 0.2,
            "tau": 0.005,
            "update_freq": 50
        },  
        "dqn": {
            "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)],
            "input_normaliser": "***TODO***",
            "gamma": "***TODO***", 
            "replay_capacity": 5e5,
            "batch_size": 256,
            "lr_Q": 1e-3,
            "epsilon_decay": "***TODO***" # P["deployment"]["num_episodes"]*P["deployment"]["episode_time_limit"],
        }
    }
}
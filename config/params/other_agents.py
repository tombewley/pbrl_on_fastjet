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
        "steve": {
            "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
            "replay_capacity": 1e6,
            "num_random_steps": 0, 
            "num_models": 2, 
            "model_freq": 1, 
            "lr_model": 1e-3, 
            "horizon": 5, 
            "ddpg_parameters": {"td3": True},
            "nonfixed_dim": 19 # NOTE: First 19 dimensions vary over an episode
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
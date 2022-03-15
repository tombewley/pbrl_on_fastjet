P = {
    "deployment": {
        "agent": "steve"
    },
    "agent": {
        "steve": {
            "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
            "replay_capacity": 5e5,
            "num_random_steps": 1e5, 
            "num_models": 2, 
            "model_freq": 1, 
            "lr_model": 1e-3, 
            "horizon": 5, 
            "ddpg_parameters": {"td3": True},
            "nonfixed_dim": 19 # NOTE: First 19 dimensions vary over an episode
        }
    }
}
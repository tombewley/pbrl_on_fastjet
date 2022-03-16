P = {
    "deployment": {
        "agent": "simple_model_based",
        "do_extra": True
    },
    "agent": {
        "simple_model_based": {  
            "net_model": [(None, 128), "R", (128, 128), "R", (128, None)],
            "probabilistic": False,
            "replay_capacity": 5e5,
            "num_random_steps": 1e5,
            "batch_size": 32,
            "model_freq": 10, # Number of steps between model updates
            "lr_model": 1e-3,
            "batch_ratio": 0.9, # Proportion of on-policy transitions
            "num_rollouts": 50,
            "rollout_horizon": 10,
            "gamma": 0.99,
        }
    }
}
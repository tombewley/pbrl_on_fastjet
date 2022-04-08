P = {
    "deployment": {
        "agent": "mbpo",
    },
    "agent": {
        # ---------------------- SAC ----------------------
        "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
        "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
        "input_normaliser": "box_bounds",
        "replay_capacity": 5e5,
        "batch_size": 32, 
        "lr_pi": 1e-4, 
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "alpha": 0.2, 
        "tau": 0.005, 
        "update_freq": 1,
        # -------------------------------------------------
        "net_model": [(None, 200), "R", (200, 200), "R", (200, 200), "R", (200, 200), "R", (200, None)],
        "ensemble_size": 7,
        "probabilistic": False,
        "num_random_steps": 0,
        "model_freq": 1,
        "batch_size_model": 256,
        "lr_model": 1e-3,
        "rollouts_per_update": 1,
        "rollout": {
            # initial, final, (start of change, end of change) in units of model updates.
            "horizon_params": ("linear", 1, 20, (2000, 10000)), 
        },
        "retained_updates": 2000,
        "policy_updates_per_timestep": 20
    }
}
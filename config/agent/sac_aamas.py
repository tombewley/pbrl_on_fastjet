P = {
    "deployment": {
        "agent": "sac",
    },
    "agent": {
        # Parameters for LunarLander, taken from AAMAS 2022 paper
        "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
        "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
        "replay_capacity": 2e5,
        "batch_size": 64, 
        "lr_pi": 1e-4, 
        "lr_Q": 1e-3,
        "gamma": 0.99, 
        "alpha": 0.2, 
        "tau": 0.01,
        "update_freq": 1 
    }
}

P = {
    "deployment": {
        "agent": "sac",
    },
    "agent": {
        "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
        "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
        "input_normaliser": "box_bounds",
        "replay_capacity": 5e4,
        "batch_size": 32, 
        "lr_pi": 1e-4, 
        "lr_Q": 1e-3,
        "gamma": 0.99, 
        "alpha": 0.2, 
        "tau": 0.005,
        "update_freq": 1 
    }
}

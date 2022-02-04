P={}

P["agent"] = {
    "sac": {
        "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
        "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
        "input_normaliser": "obs_lims",
        "replay_capacity": 5e5,
        "batch_size": 32, 
        "lr_pi": 1e-4, 
        "lr_Q": 1e-3,
        "gamma": 0.75, 
        "alpha": 0.2, 
        "tau": 0.005, 
        "update_freq": 1, 
    }
}
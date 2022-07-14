P = {
    "deployment": {
        "agent": "dqn",
    },
    "agent": {
        "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)],
        "replay_capacity": 2e5,
        "batch_size": 64,
        "lr_Q": 1e-3,
        "gamma": 0.99,
        "epsilon_start": 0.9,
        "epsilon_end": 0.05,
        "epsilon_decay": 200*200,
        "target_update": ("soft", 0.005),
    }
}
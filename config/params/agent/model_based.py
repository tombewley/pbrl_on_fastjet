P = {
    "deployment": {
        "agent": "simple_model_based",
    },
    "agent": {
        "simple_model_based": {  
            "net_model": [(None, 128), "R", (128, 128), "R", (128, None)],
            "input_normaliser": "box_bounds",
            "probabilistic": False,

            "ensemble_size": 5,

            "replay_capacity": 50000,
            "num_random_steps": 5000,
            "batch_ratio": 0.9,

            "model_freq": 1,
            "batch_size": 64,
            "lr_model": 1e-3,

            "planning": {
                "num_iterations": 10,
                "num_particles": 100,
                "horizon": 20,
                "num_elites": 10,
                "alpha": 0.1,
                "gamma": 0.99
            }
        }
    }
}
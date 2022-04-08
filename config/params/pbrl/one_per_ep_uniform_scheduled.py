P = {
    "deployment": {
        "num_episodes": 200
    },
    "pbrl": {
        "feedback_budget": 10000,
        "observe_freq": 1, 
        "feedback_freq": 1,
        "num_episodes_before_freeze": 200,
        "scheduling_coef": 1,
        "sampler": {
            "weight": "uniform",
            "constrained": False,
            "probabilistic": True
        }
    }
}
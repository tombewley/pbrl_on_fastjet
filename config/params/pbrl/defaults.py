P = {
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 50, 
        "feedback_freq": 100,
        "num_episodes_before_freeze": 50000, 
        "scheduling_coef": 0,
        "sampler": {
            "weight": "ucb", 
            "constrained": True, 
            "probabilistic": True, 
            "num_std": 0 # NOTE: <<<<
        }
    }
}
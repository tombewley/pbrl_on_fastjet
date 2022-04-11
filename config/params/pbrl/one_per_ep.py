P = {
    "deployment": {
        "num_episodes": 200,
        "checkpoint_freq": 50,
    },
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 1,
        "feedback_freq": 1,
        "num_episodes_before_freeze": 200,
        "scheduling_coef": 0,
        "sampler": {
            "weight": "ucb", 
            "num_std": 0, # NOTE: <<<<
            "recency_constraint": True,
            "probabilistic": True
        },
        "save_freq": 50
    }
}
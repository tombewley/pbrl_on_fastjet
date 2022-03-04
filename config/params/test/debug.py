P = {
    "deployment": {
        "num_episodes": 40
    },
    "pbrl": {
        "feedback_budget": 40,
        "observe_freq": 1,
        "feedback_freq": 2,
        "num_episodes_before_freeze": 40,
        "scheduling_coef": 0,
        "sampler": {
            "weight": "ucb", 
            "constrained": True, 
            "probabilistic": True, 
            "num_std": 0 # NOTE: <<<<
        },
        "logger": {
            "freq": 2,
            "plots": {
                "tree_rectangles": {
                    (((2, 3), None))
                }
            }
        }
    }
}
P = {
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 1, 
        "feedback_freq": 1,
        "num_episodes_before_freeze": 1000, 
        "scheduling_coef": 0,
        "sampler": {
            "weight": "ucb", 
            "constrained": True, 
            "probabilistic": True, 
            "num_std": 0 # NOTE: <<<<
        },
        # "explainer": {
        #     "plots": [
        #         # "preference_matrix",
        #         "alignment"
        #     ],
        #     "freq": 10,
        #     "save": False,
        #     "show": True
        # },
        "save_freq": 20
    }
}
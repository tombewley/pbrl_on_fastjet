P = {
    "deployment": {
        "num_episodes": 40,
        "checkpoint_freq": 40
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
        "save_freq": 40
        # "explainer": {
        #     "freq": 2,
        #     "plots": {
        #         "tree_loss_vs_m"
        #     }
        # }
    }
}
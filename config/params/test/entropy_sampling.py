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
            "weight": "entropy", 
            "constrained": False, 
            "probabilistic": True,
            "preference_eqn": "thurstone"
        },
        "explainer": {
            "freq": 2,
            "plots": {
                "sampler_matrix"
            }
        }
    }
}
P = {
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 50, 
        "num_episodes_before_freeze": 50000, 
        "scheduling_coef": 0,
        "sampler": {
            "weight": "entropy", 
            "constrained": False, 
            "probabilistic": True,
            "preference_eqn": "thurstone"
        }
    }
}
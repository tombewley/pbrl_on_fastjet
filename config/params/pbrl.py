P = {
    "pbrl": {
        "reward_source": "tree",  
        # "feedback_budget": 10000, # Disable for sweep.
        # "observe_freq": 50, 
        # "feedback_freq": 500, # By ep not n
        # "log_freq": 500, # By ep not n      
        # "num_episodes_before_freeze": 50000, 
        # "scheduling_coef": 0,
        "sampling": {
            "weight": "ucb", 
            "constrained": True, 
            "probabilistic": True, # NOTE: Very early indication that this may be better.
            "num_std": 2
            },
        "p_clip": 0.1,
        "m_max": 100,
        "m_stop_merge": 1, 
        "min_samples_leaf": 1,
        "alpha": 0.001,
    }
}
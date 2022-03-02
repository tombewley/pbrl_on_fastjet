P = {
    "pbrl": {
        "reward_model": "net",  
        "sampling": {
            "weight": "ucb", 
            "constrained": True, 
            "probabilistic": True, 
            "num_std": 0 # NOTE: <<<<
            },
        "feedback_budget": 10000, 
        "scheduling_coef": 0,
        "observe_freq": 50, 
        "feedback_freq": 100, 
        "num_episodes_before_freeze": 50000, 
        
        "num_batches_per_update": 100,
        "batch_size": 32
    }
}
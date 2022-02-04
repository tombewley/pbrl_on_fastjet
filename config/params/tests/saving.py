P={}

N = 30

P["deployment"] = {
    "num_episodes": N,
    "checkpoint_freq": 0,
}

P["pbrl"] = {
    "feedback_budget": N*10,
    "observe_freq": 1, 
    "feedback_freq": 1, 
    "num_episodes_before_freeze": N, 
    "save_freq": N, 

    "reward_source": "tree",  
    "scheduling_coef": 0,
    "sampling": {
        "weight": "ucb", 
        "constrained": True, 
        "probabilistic": True, 
        "num_std": 2
        },
    "p_clip": 0.1,
    "m_max": 100,
    "m_stop_merge": 1, 
    "min_samples_leaf": 1,
    "alpha": 0.005,
}
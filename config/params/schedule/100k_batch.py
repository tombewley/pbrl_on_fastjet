P = {
    "deployment": {
        "num_episodes": 100000,
        "checkpoint_freq": 25000
    },
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 50, 
        "feedback_freq": 100,
        "num_episodes_before_freeze": 50000, 
        "scheduling_coef": 0,
        "save_freq": 25000
    }
}

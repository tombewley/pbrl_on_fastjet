P = {
    "deployment": {
        "num_episodes": 1000,
        "train": False,
        # "checkpoint_freq": 50
    },
    "pbrl": {
        "feedback_budget": 10000, 
        "observe_freq": 1,
        "feedback_freq": 10,
        "num_episodes_before_freeze": 1000,
        "scheduling_coef": 0,
        "save_freq": 50
    }
}

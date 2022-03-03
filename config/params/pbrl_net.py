from rlutils.observers.models import RewardNet

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "kind": RewardNet,
            "preference_eqn": "thurstone",
            "num_batches_per_update": 100,
            "batch_size": 32
        },
        "sampler": {
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
    }
}
from rlutils.observers.pbrl.models import RewardTree

P = {
    "pbrl": {
        "model": {
            "kind": RewardTree,
            "p_clip": 0.1,
            "m_max": 100,
            "m_stop_merge": 1, 
            "min_samples_leaf": 1,
            "alpha": 0.001,
        },
        # "feedback_freq": 500 
    }
}
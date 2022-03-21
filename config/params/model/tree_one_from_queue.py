from rlutils.observers.pbrl.models import RewardTree

P = {
    "pbrl": {
        "model": {
            "kind": RewardTree,
            "preference_eqn": "thurstone",
            "loss_func": "bce",
            "split_by_variance": False,
            "p_clip": 0.1,
            "m_max": 100,
            "num_from_queue": 1,
            "min_samples_leaf": 1,
            "store_all_qual": False,
            "alpha": 0.001,
        },
        "feedback_freq": 500,
        "reward_source": "model"
    }
}
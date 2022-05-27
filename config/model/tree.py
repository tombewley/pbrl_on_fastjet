from rlutils.observers.pbrl.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": RewardTree,
            # "preference_eqn": "thurstone",
            "loss_func": "bce",
            "p_clip": 0.1,
            "m_max": 100,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "store_all_qual": False,
        }
    }
}
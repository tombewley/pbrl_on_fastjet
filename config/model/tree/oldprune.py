from rlutils.rewards.pbrl.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": RewardTree,

            "preference_eqn": "bradley-terry",
            "loss_func": "bce",
            "trees_per_update": 1,
            "prune_ratio": None,
            "alpha": 0.001,

            "m_max": 100,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "store_all_qual": False,
        }
    }
}
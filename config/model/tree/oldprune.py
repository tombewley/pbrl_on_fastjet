from rlutils.rewards.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": RewardTree,

            "preference_eqn": "bradley-terry",
            "loss_func": "0-1",
            "trees_per_update": 1,
            "prune_ratio": None,
            "alpha": 0.01, # 0.001,

            "m_max": 100,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "store_all_qual": False,
        }
    }
}
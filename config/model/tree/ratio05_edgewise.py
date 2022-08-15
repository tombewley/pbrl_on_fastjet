from rlutils.rewards.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": RewardTree,

            "preference_eqn": "bradley-terry",
            "loss_func": "bce",
            "trees_per_update": 5,
            "prune_ratio": 0.5,
            "nodewise_partition": False,
            "post_populate_with_all": False,
            "alpha": 0,

            "m_max": 100,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "store_all_qual": False,
        }
    }
}
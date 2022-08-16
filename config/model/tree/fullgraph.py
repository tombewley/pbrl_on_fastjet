from rlutils.rewards.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": RewardTree,
            "trees_per_update": 1,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "preference_eqn": "bradley-terry",
            "prune_ratio": None,
            "store_all_qual": False,
            "m_max": 100,
            #             0       1       2       3       4       5
            "loss_func": ["0-1",  "0-1",  "0-1",  "bce",  "bce",  "bce" ],
            "alpha":     [0.,     0.001,  0.01,   0.,     0.001,  0.01  ]
        }
    }
}
from rlutils.observers.pbrl.models import RewardTree

P = {
    "pbrl": {
        "model": {
            "kind": RewardTree,
            "split_by_variance": True,
            "p_clip": 0.1,
            "m_max": 100,
            "num_from_queue": float("inf"), # NOTE: <<<
            "min_samples_leaf": 1,
            "store_all_qual": False,
            "alpha": 0.001,
        },
        "feedback_freq": 500
    }
}
from ...features import preprocessor, features
from rlutils.observers.pbrl.models import RewardTree

P = {
    "pbrl": {
        "featuriser": {
            "preprocessor": preprocessor,
            "features": features
        },
        "model": {
            "class": RewardTree,
            "preference_eqn": "thurstone",
            "loss_func": "0-1",
            "split_by_variance": True,
            "p_clip": 0.1,
            "m_max": 100,
            "num_from_queue": float("inf"),
            "min_samples_leaf": 1,
            "store_all_qual": False,
            "alpha": 0.001,
        },
        "reward_source": "model"
    }
}
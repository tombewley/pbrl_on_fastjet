from ...features import preprocessor, features
from rlutils.observers.pbrl.models import RewardNet

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "featuriser": {
                "preprocessor": preprocessor,
                "features": features
            },
            "class": RewardNet,
            "preference_eqn": "bradley-terry",
            "num_batches_per_update": 100,
            "batch_size": 32
        }
    }
}
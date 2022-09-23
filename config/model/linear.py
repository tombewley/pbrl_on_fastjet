from rlutils.rewards.models import LinearRewardModel

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class": LinearRewardModel,
            "preference_eqn": "bradley-terry",
            "num_batches_per_update": 100,
            "batch_size": 32
        }
    }
}
from rlutils.observers.pbrl.models import RewardNet

P = {
    "pbrl": {
        "model": {
            "kind": RewardNet,
            "preference_eqn": "bradley-terry",
            "num_batches_per_update": 100,
            "batch_size": 32
        },
        # "feedback_freq": 100
    }
}
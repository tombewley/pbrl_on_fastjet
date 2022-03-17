from rlutils.observers.pbrl import load
from rlutils.observers.pbrl.models import RewardNet, RewardTree
from rlutils.observers.pbrl.interfaces import OracleInterface
from config.interface import FastJetInterface

from config.features import F
from config.oracles import *

P = {
    "model": {
        "kind": RewardTree,
        "split_by_variance": True,
        "p_clip": 0.1,
        "m_max": 10,
        "num_from_queue": float("inf"),
        "min_samples_leaf": 1,
        "alpha": 0.001,
        "store_all_qual": False
        },
    "sampler": {
        "weight": "uniform", 
        "constrained": False,
        "probabilistic": True
    },
    "interface": {
        "kind": OracleInterface, 
        "oracle": dist_closing_uperr
    }
}

pbrl = load(f"logs/follow_random/1000.pbrl", P, features=F)
pbrl.sampler.seed(0)

pbrl.preference_batch(10)
pbrl.update("test")

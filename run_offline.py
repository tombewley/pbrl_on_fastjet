from rlutils.observers.pbrl import load
from rlutils.observers.pbrl.models import RewardNet, RewardTree
from rlutils.observers.pbrl.interfaces import OracleInterface
from fastjet.interface import FastJetInterface

from config.features import F
from config.oracles import *

P = {
    "model": {
        "kind": RewardTree,
        "p_clip": 0.1,
        "m_max": 100,
        "m_stop_merge": 1, 
        "min_samples_leaf": 1,
        "alpha": 0.001,
        },
    "sampler": {
        "weight": "uniform", 
        "constrained": True,
        "probabilistic": True
    },
    "interface": {
        "kind": OracleInterface, 
        "oracle": target_pose_tree
    }
}

pbrl = load(f"logs/2022-03-04_16-30-00/100.pbrl", P, features=F)

pbrl.preference_batch(100)
pbrl.update("test")

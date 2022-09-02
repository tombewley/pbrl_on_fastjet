from torch import load
from numpy import mean
from rlutils.rewards.models import RewardTree
from config.features.fastjet.default import P as P_featuriser

from torch import set_printoptions
set_printoptions(precision=12)

graph = load("offline_graphs/fastjet/match/dist_pose_when_close/0_200e_1000p.graph")

model = RewardTree({
    "featuriser": P_featuriser["pbrl"]["model"]["featuriser"],
    "preference_eqn": "bradley-terry",
    "loss_func": "0-1",
    "min_samples_leaf": 1,
    "split_dim_entropy": 0,
    "num_from_queue": float("inf"),
    "alpha": 0,

    "trees_per_update": 1,
    "prune_ratio": None,
    "nodewise_partition": False,
    "m_max": 100,
    "split_by_preference": True,
    "store_all_qual": True,
})

model.seed(1)

model.update(graph, mode="preference", history_key=None)

from hyperrectangles.visualise import *
import matplotlib.pyplot as plt
show_split_quality(model.forest[0].root, figsize=(20,20))
plt.show()

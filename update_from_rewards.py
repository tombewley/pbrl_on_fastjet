from torch import load
from rlutils.rewards.interfaces import OracleInterface
from rlutils.rewards.models import RewardNet, RewardTree
from config.features.fastjet.follow.default import P

graph = load("offline_graphs/fastjet/follow/dist_closing_uperr_v2/0_100e_4950p.graph")

# graph._graph.remove_edges_from(list(graph.edges))
# graph.add_preference(4, 6, 0.1)

print(graph)

P = P["pbrl"]["model"]

if False:
    P.update({"batch_size": 32, "num_batches_per_update": 1000, "preference_eqn": "bradley-terry"})
    model = RewardNet(P)
else:
    P.update({"trees_per_update": 1, "prune_ratio": None, "m_max": 100, "min_samples_leaf": 1, "num_from_queue": float("inf"), "store_all_qual": False,
              "preference_eqn": "bradley-terry", "loss_func": "bce", "alpha": 0.001})
    model = RewardTree(P)

print(model.update(graph, mode="reward", history_key="test"))

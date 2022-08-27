import os
from torch import load
import matplotlib.pyplot as plt
from rlutils.rewards.evaluate import *
from rlutils.rewards.epic import graph as epic_graph, draw_graph
from config.oracle.fastjet.follow.dist_closing_uperr_v2 import oracle


run_names = [
    "fastjet/follow/dist_closing_uperr_v2/hopeful-forest-118",
    "fastjet/follow/dist_closing_uperr_v2/rosy-planet-119",
    "fastjet/follow/dist_closing_uperr_v2/rose-plasma-120",
    "fastjet/follow/dist_closing_uperr_v2/sunny-planet-123",
    "fastjet/follow/dist_closing_uperr_v2/hearty-sea-124"
]

graph = load(
    "offline_graphs/fastjet/follow/dist_closing_uperr_v2/0_100e_4950p.graph"
)

reward_functions = []
edgelist = []
n = 0
for run_name in run_names:
    steps = sorted([x for x in os.listdir(f"graphs_and_models/{run_name}")
        if x[-7:] == ".reward"], key=lambda x:int(x.split(".")[0]))
    print(steps)
    for i, step in enumerate(steps):
        reward_functions.append(load(f"graphs_and_models/{run_name}/{step}"))
        if i > 0: edgelist.append((n-1, n))
        n += 1

# tau = rank_correlation(graph, reward_functions + [oracle])
# print(tau[-1,:-1])
# loss = preference_loss(graph, reward_functions, loss_func="0-1")
# print(loss)
corr_r, corr_g, _, _ = epic(graph, reward_functions + [oracle], num_canon=0)
print(corr_r[-1,:-1])

plt.figure()
draw_graph(epic_graph(corr_r), edgelist=edgelist)
plt.show()

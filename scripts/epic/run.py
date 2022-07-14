import os
from torch import load as pt_load, vstack, cat
import networkx as nx
import matplotlib.pyplot as plt
from rlutils.observers.pbrl import load
from rlutils.rewards.epic import epic, mds_layout
from config.oracle.fastjet.follow.dist_closing_uperr_v2 import oracle

def oracle_wrapped(states, actions, next_states):
    return oracle(cat([states, actions, next_states], dim=-1))

names = []
reward_functions = []
edgelist = []
n = 0
for run_name in ["fanciful-shadow-109","woven-deluge-110","devout-jazz-111"]:
    steps = sorted(os.listdir(f"models/{run_name}"), key=lambda x:int(x.split(".")[0]))
    for i, step in enumerate(steps):
        names.append(f"{run_name}/{step.split('.')[0]}")
        reward_functions.append(load(f"models/{run_name}/{step}", {"reward_source": "model"}).reward)
        if i > 0: edgelist.append((n-1, n))
        n += 1
names.append("oracle")
reward_functions.append(oracle_wrapped)

graph = pt_load("epic/datasets/follow/dist_closing_uperr_v2.graph")
transitions = vstack([ep["transitions"] for _,ep in graph.nodes(data=True)])
states = transitions[:,:37]
actions = transitions[:,37:41]
next_states = transitions[:,-37:]

# NOTE:
canon_actions, canon_next_states = actions[:100], next_states[:100]

g = epic(reward_functions, states, actions, next_states, canon_actions, canon_next_states)
plt.figure()
nx.draw(g, pos=mds_layout(g), node_size=0, with_labels=True, edgelist=edgelist)

plt.show()

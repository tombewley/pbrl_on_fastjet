"""
Generate MDS scatter plots for reward functions based on a similarity metric,
colouring the point for each function by its oracle regret.
"""
import os
import argparse
from torch import device, load
from torch.cuda import is_available
from numpy import percentile
from networkx import draw
import matplotlib.pyplot as plt
from rlutils import build_params
from rlutils.rewards.evaluate import *


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("--metric", type=str, default="rank_correlation")
parser.add_argument("--invert_colours", type=int, default=0)
args = parser.parse_args()

COLOURS = {
    "oracle": "w" if args.invert_colours else "k",
    "net": "#99544eff",
    "tree_var": "#5b5699ff",
    "tree_pref": "#23995eff"
}

device_ = device("cuda" if is_available() else "cpu")
P = build_params([f"oracle.fastjet.{args.task}.{args.oracle}"], root_dir="config")
oracle = P["pbrl"]["interface"]["oracle"]
preference_graph = load(P["pbrl"]["offline_graph_path"], map_location=device_)
preference_graph.device = device_

rollout_graphs = load(f"final_model_rollout_graphs/fastjet/{args.task}/{args.oracle}/500e_0p.graphs", map_location=device_)
lo =  percentile(rollout_graphs["random"].oracle_returns, 50)
rng = percentile(rollout_graphs["oracle"].oracle_returns, 50) - lo

path = f"final_graphs_and_models/fastjet/{args.task}/{args.oracle}"
reward_functions, names, regret_reductions = [oracle], ["oracle"], [1.]
for subdir in os.listdir(path):
    for run in os.listdir(f"{path}/{subdir}"):
        name = f"{subdir}/{run}"
        model = load(f"{path}/{name}/0200.reward", map_location=device_)
        model.device = device_
        reward_functions.append(model)
        names.append(name)
        regret_reductions.append((percentile(rollout_graphs[subdir][run].oracle_returns, 50) - lo) / rng)

if args.metric == "reward_correlation":
    corr, _, _, _ = epic(preference_graph, reward_functions)
elif args.metric == "return_correlation":
    _, corr, _, _ = epic(preference_graph, reward_functions)
elif args.metric == "rank_correlation":
    corr = rank_correlation(preference_graph, reward_functions)
else: raise NotImplementedError

for i, name in enumerate(names):
    print(f"{name.ljust(40)}: {corr[i,0]}")

g = graph(corr_to_dist(corr))
pos = mds_graph_layout(g)

plt.figure(figsize=(0.7,0.7))
draw(g,
     pos=pos,
     edgelist=[],
     node_size=6,
     node_color=regret_reductions,
     cmap="Greys" + ("" if args.invert_colours else "_r"),
     vmin=0,
     vmax=1
)
draw(g,
     pos=pos,
     edgelist=[],
     node_size=0.5,
     node_color=[COLOURS[n.split("/")[0]] for n in names],
     # labels={i:n for i,n in enumerate(names)}
)
# plt.show()
plt.savefig(f"mds_{args.task}_{args.metric}.svg")

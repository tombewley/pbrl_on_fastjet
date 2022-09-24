"""
Visualise a tree-structured reward function as a diagram.
"""
import os
import argparse
from torch import load
from torch.cuda import is_available
from numpy import zeros
from rlutils.common.featuriser import Featuriser
from config.features.fastjet import *
from hyperrectangles.rules import diagram
from hyperrectangles.visualise import show_rectangles, show_episodes
import tufte
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("run", type=str)
parser.add_argument("--plt", type=int, default=0)
args = parser.parse_args()

os.chdir(f"final_graphs_and_models/fastjet/{args.task}/{args.oracle}/{args.run}")
model = load("0200.reward", map_location=device); model.device = device  # NOTE: Final only
tree = model.forest[0]["tree"]
graph = load("0200_200e_1000p.graph", map_location=device); graph.device = device
featuriser = Featuriser({
    "preprocessor": preprocessor,
    "features": [dist, alt]
})

# Rename features for visualisation
f = tree.space.dim_names
f = [ff.replace("_", " ") for ff in f]
f[-1] = "r"
tree.space.dim_names = f

# Gather trajectory-level return estimates (divided by lengths) and use for cmap_lims
eps, rewards = tree.root.data("ep", "r").T
g_per_t = zeros(200)
for i, r in zip(eps.astype(int), rewards): g_per_t[i] = r
cmap_lims = (min(g_per_t), max(g_per_t))

# Sense check that tree's ep nums line up with graph
if False: show_episodes(tree.space, ("dist", "alt"), ep_indices)

# Sense check that g_per_t aligns with new maximum likelihood result
if False:
    _, _, _, A, _, _, y = graph.preference_data_structures()
    plt.figure(); plt.scatter(model.maximum_likelihood_returns(A, y)[0], g_per_t, s=20)

# Make diagram
def do_diagram(tree_):
    diag = diagram(tree_, pred_dims=["r"], colour_dim="r", cmap_lims=cmap_lims,
                   show_num_samples=True, show_decision_node_preds=True, out_as="plt" if args.plt else "svg")
    if args.plt:
        _, ax = plt.subplots(figsize=(30,30)); ax.axis("off")
        ax.imshow(diag)
do_diagram(tree)

# =============================================================

# TODO: Split out

subtree = tree.dca_subtree("dist_alt_subtree", {
    tree.root.left.left,
    tree.root.left.right,
    tree.root.right.left.left,
    tree.root.right.left.right.left,
    tree.root.right.left.right.right,
    tree.root.right.right
})
# do_diagram(subtree)

vis_lims = ((0, 150), (0, 350)) # To match Figure 4
ax = tufte.ax(
    w_inches=1.150-0.084,
    h_inches=1.150-0.084,
)
show_rectangles(subtree, ("dist", "alt"), attribute=("mean", "r"),
                ax=ax, cmap_lims=cmap_lims, vis_lims=vis_lims, edge_colour="k", cbar=False)
ax.spines["top"].set_visible(True)
ax.spines["right"].set_visible(True)
ax.set_xlim(*vis_lims[0])
ax.set_ylim(*vis_lims[1])
ax.save("subtree_partition.svg")

tufte.line_width(0.260)
ax = tufte.ax(
    w_inches=1.150-0.084,
    h_inches=1.150-0.084,
    x_ticks=(),
    y_ticks=()
)
features = [featuriser(s, a, ns) for s, a, ns in zip(graph.states, graph.actions, graph.next_states)]
tufte.coloured_2d_plot(ax, features,
    colour_by=g_per_t,
    cmap="coolwarm_r",
    cmap_lims=cmap_lims,
    alpha=1
)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_xlim(*vis_lims[0])
ax.set_ylim(*vis_lims[1])
ax.save("trajectories.svg")

# print(subtree.root.left.left)

plt.show()

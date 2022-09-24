"""
Plot rollout trajectories for a model, either as 1D time series, or in a 2D feature subspace.
"""
import argparse
from torch import load
from numpy import percentile
from math import pi
import tufte
from rlutils import build_params
from rlutils.common.featuriser import Featuriser
from config.features.fastjet import *


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("model", type=str)
parser.add_argument("f0", type=str)
parser.add_argument("--f1", type=str)
parser.add_argument("--x_min", type=float, default=None)
parser.add_argument("--x_max", type=float, default=None)
parser.add_argument("--y_min", type=float, default=None)
parser.add_argument("--y_max", type=float, default=None)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--axis", type=int, default=0)
args = parser.parse_args()

print("do best by offline rank correlation?")
best_runs_by_orr = { # By oracle regret ratio (rollout_final_models.py)
    "match": {
        "net":          "usual-voice-871",
        "tree_var":     "flowing-frog-896",
        "tree_pref":    "eager-river-873"
    },
    "follow": {
        "net":          "genial-frog-874",
        "tree_var":     "neat-wood-827",
        "tree_pref":    "charmed-hill-900"
    },
    "chase": {
        "net":          "jolly-pine-1559",
        "tree_var":     "confused-serenity-1543",
        "tree_pref":    "resilient-oath-1523"
    },
    "land": {
        "net":          "ruby-cloud-1417",
        "tree_var":     "skilled-mountain-1266",
        "tree_pref":    "playful-night-1258"
    }
}
best_runs_by_rank_corr = { # By offline rank correlation (plot_mds.py)
    "match": {
        "net":          "summer-dust-823",
        "tree_var":     "flowing-frog-896",
        "tree_pref":    "avid-leaf-801"
    },
    "follow": {
        "net":          "effortless-cherry-778",
        "tree_var":     "neat-wood-827",
        "tree_pref":    "sunny-plant-876"
    },
    "chase": {
        "net":          "jolly-pine-1559",
        "tree_var":     "tough-surf-1521",
        "tree_pref":    "resilient-oath-1523"
    },
    "land": {
        "net":          "giddy-tree-1397",
        "tree_var":     "soft-silence-1269",
        "tree_pref":    "ruby-wood-1261"
    }
}

graphs = load(f"final_model_rollout_graphs/fastjet/{args.task}/{args.oracle}/500e_0p.graphs", map_location=device)
cmap_lims = (percentile(graphs["random"].oracle_returns, 50), percentile(graphs["oracle"].oracle_returns, 50))
featuriser = Featuriser({
    "preprocessor": preprocessor,
    "features": [locals()[args.f0]] + ([locals()[args.f1]] if args.f1 is not None else [])
})

if args.model in {"random", "oracle"}:
    graph = graphs[args.model]
    reward_function = build_params([f"oracle.fastjet.{args.task}.{args.oracle}"], root_dir="config")["pbrl"]["interface"]["oracle"]
else:
    run = best_runs_by_orr[args.task][args.model]
    graph = graphs[args.model][run]
    reward_function = load(f"final_graphs_and_models/fastjet/{args.task}/{args.oracle}/{args.model}/{run}/0200.reward", map_location=device)
    reward_function.device = device

features = [featuriser(s, a, ns) for s, a, ns in zip(graph.states, graph.actions, graph.next_states)]
tufte.line_width(0.130)
ax = tufte.ax(
    w_inches=0.472,
    h_inches=0.472,
)
tufte.coloured_2d_plot(ax, features, colour_by=graph.oracle_returns,
                       cmap_lims=cmap_lims, cmap="coolwarm_r", alpha=args.alpha,
                       x_axis_offset=1
                       )

if args.model != "random":
    # Highlight the single trajectory that the model thinks has highest return by plotting it in black
    _, [returns] = graph.rewards_by_ep_and_returns([reward_function])
    best_ep = features[returns.argmax()]
    if args.f1 is None:
        ax.plot(range(1, len(best_ep)+1), best_ep[:,0], color="k", lw=0.130*3, ls="--")
    else:
        ax.plot(best_ep[:,0], best_ep[:,1], color="k", lw=0.130*3, ls="--")

ax.set_xlim(args.x_min, args.x_max)
ax.set_ylim(args.y_min, args.y_max)
ax.set_xticks([ax.get_xlim()[0], ax.get_xlim()[1]])
ax.set_yticks([ax.get_ylim()[0], 0, pi / 12, ax.get_ylim()[1]])
print(ax.get_xlim())
print(ax.get_ylim())

# ax.show()

if args.axis: ext = "svg"
else: ext = "png"; ax.axis("off")
ax.save(f"{args.task}-{args.oracle}-{args.model}_{args.f0}-{args.f1}.{ext}", dpi=700)

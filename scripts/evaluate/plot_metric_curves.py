"""
Generate publication-ready time series plots of fidelity and performance metrics.
"""
import os
import argparse
from torch import device, load
from torch.cuda import is_available
from numpy import percentile
import tufte
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("load_path", type=str)
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("metric", type=str)
# parser.add_argument("num_episodes", type=int)
parser.add_argument("--coef", type=int, default=0)
parser.add_argument("--y_min", type=float, default=None)
parser.add_argument("--y_max", type=float, default=None)
parser.add_argument("--step", type=int, default=1)
args = parser.parse_args()

MODELS = {
    # f"pbrl.model.class=rlutils.rewards.models.RewardNet-num_episodes={args.num_episodes}-": {
    f"pbrl.model.class=rlutils.rewards.models.RewardNet-pbrl.scheduling_coef={args.coef}": {
        "label": "NN",
        "colour": "#99544eff"
    },
    # f"pbrl.model.class=rlutils.rewards.models.RewardTree-pbrl.model.split_by_preference=False-num_episodes={args.num_episodes}-": {
    f"pbrl.model.class=rlutils.rewards.models.RewardTree-pbrl.model.split_by_preference=False-pbrl.scheduling_coef={args.coef}": {
        "label": "Tree (var)",
        "colour": "#5b5699ff"
    },
    # f"pbrl.model.class=rlutils.rewards.models.RewardTree-pbrl.model.split_by_preference=True-num_episodes={args.num_episodes}-": {
    f"pbrl.model.class=rlutils.rewards.models.RewardTree-pbrl.model.split_by_preference=True-pbrl.scheduling_coef={args.coef}": {
        "label": "Tree (0-1)",
        "colour": "#23995eff"
    }
}

if args.metric == "oracle_return":
    graphs_path = f"final_model_rollout_graphs/fastjet/{args.task}/{args.oracle}/500e_0p.graphs"
    graphs = load(graphs_path, map_location=device("cuda" if is_available() else "cpu"))
    hi = percentile(graphs["oracle"].oracle_returns, 50)
    rng =  hi - percentile(graphs["random"].oracle_returns, 50)

tufte.line_width(0.369)
ax = tufte.ax(
    w_inches=0.661,
    h_inches=0.7,
    x_ticks=(),#0, args.num_episodes-1)
)

os.chdir(f"results/{args.load_path}/{args.task}")
for fname in os.listdir():
    if f"{args.metric}.csv" in fname:
        _m = [m for m in MODELS if m in fname]
        if not _m: continue
        model = MODELS[_m[0]]
        print(fname)
        data = pd.read_csv(fname).values
        if args.metric == "oracle_return":
            data = (hi - data) / rng
        elif args.metric == "_runtime":
            data = data / 60

        tufte.smoothed_time_series(ax,
            data=data,
            step=args.step,

            # radius=1,
            # q=(),

            radius=args.step * 10,
            q=(25,),

            colour=model["colour"],
            ls=":" if "online" in fname else "-",
            shade_alpha=0.25,
        )
ax.set_ylim(args.y_min, args.y_max)

# ax.show()
ax.save(f"{args.task}_{args.metric}.svg")

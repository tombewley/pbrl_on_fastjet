"""
Generate publication-ready time series plots of fidelity and performance metrics.
"""
import os
import argparse
from torch import device, load
from torch.cuda import is_available
from numpy import percentile, nanpercentile
import tufte
import pandas as pd
from matplotlib import rcParams
rcParams['lines.markersize'] = 1.5


parser = argparse.ArgumentParser()
parser.add_argument("load_path", type=str)
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("y_metric", type=str)
parser.add_argument("x_metric", type=str)
parser.add_argument("--x_min", type=float, default=0)
parser.add_argument("--x_max", type=float, default=1)
parser.add_argument("--ticks", type=int, default=0)
parser.add_argument("--descending", type=int, default=0)
args = parser.parse_args()

MODELS = {
    f"pbrl.model.class=rlutils.rewards.models.RewardTree-pbrl.model.split_by_preference=True": {
        "label": "Tree (0-1)",
        "colour": "#23995eff"
    },
    f"pbrl.model.class=rlutils.rewards.models.RewardNet": {
        "label": "NN",
        "colour": "#99544eff"
    }
}

assert args.x_metric in {"oracle_return", "offline_rank_correlation"}

if args.x_metric == "oracle_return":
    graphs_path = f"final_model_rollout_graphs/fastjet/{args.task}/{args.oracle}/500e_0p.graphs"
    graphs = load(graphs_path, map_location=device("cuda" if is_available() else "cpu"))
    hi = percentile(graphs["oracle"].oracle_returns, 50)
    rng =  hi - percentile(graphs["random"].oracle_returns, 50)

tufte.line_width(0.369)
ax = tufte.ax(
    w_inches=0.584,
    h_inches=0.501,
    x_ticks=None if args.ticks else (),
    y_ticks=(),
)
# if not args.bottom: ax.spines["bottom"].set_visible(False)

os.chdir(f"results/{args.load_path}/{args.task}")
results = {model: {} for model in MODELS}
for fname in os.listdir():
    if f"{args.x_metric}.csv" in fname:
        _m = [m for m in MODELS if m in fname]
        if not _m: continue
        model = _m[0]
        print(fname)
        data = pd.read_csv(fname)
        if args.x_metric == "oracle_return":

            assert data.shape[0] == 200 and data.shape[1] >= 4, data.shape # NOTE: Need at least four runs
            nans = data.isna().any()
            if nans.any():
                print(nans)
                raise Exception

            data = (hi - data) / rng
        print(data.median())

        x_value = float(fname[fname.rfind(args.y_metric)+len(args.y_metric)+1:].split("-")[0])
        results[model][x_value] = nanpercentile(data.values, [25, 50, 75])

for i, model in enumerate(MODELS):

    print(model)
    print(results[model])
    results_sorted = [results[model][x] for x in sorted(results[model].keys(), reverse=args.descending)]
    print(results_sorted)

    if False:
        ax.barh(
            y=[yy+i*0.4 for yy in range(len(results_sorted))],
            width=results_sorted,
            height=0.4,
            align="edge",
            color=MODELS[model]["colour"]
        )
    else:
        for plot_func in (ax.scatter, ax.plot):
            plot_func(
                [m for _,m,_ in results_sorted],
                range(len(results_sorted)),
                color=MODELS[model]["colour"],
                lw=0.369
            )
        ax.fill_betweenx(
            range(len(results_sorted)),
            [l for l,_,_ in results_sorted],
            [u for _,_,u in results_sorted],
            color=MODELS[model]["colour"],
            alpha=0.25, lw=0, zorder=-1
        )

ax.set_xlim(args.x_min, args.x_max)
marg=0.65
ax.set_ylim(-marg, len(results_sorted)-1+marg)

# ax.show()
ax.save(f"{args.task}_{args.y_metric}_{args.x_metric}.svg")

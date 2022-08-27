import os
import argparse
import tufte
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("load_path", type=str)
parser.add_argument("metric", type=str)
args = parser.parse_args()

ax = tufte.ax(
    w_inches=0.5,
    w_inches=2, 
    h_inches=1, 
    x_label="Episode Number", 
    y_label=args.metric,
    # x_lims=(0, 100),
    # y_lims=(-0.2,1),
    # y_ticks=(0.4,1.0)
)

os.chdir(f"results/{args.load_path}")
if args.metric == "oracle_return":
    for fname in os.listdir():
        if f"---{args.metric}.csv" in fname and "pbrl.model.class=None" in fname:
            oracle_baseline_return = pd.read_csv(fname).mean().values.mean()
for fname in os.listdir():
    if f"---{args.metric}.csv" and "pbrl.model.class=None" not in fname:
        data = pd.read_csv(fname).values
        
        if args.metric == "oracle_return":
            data = oracle_baseline_return - data
        
        print(data)


# if False:
#     raw_data = pd.read_csv("experiments/tree_growth/test.method_agent=sac__runtime.csv").values
#     if False:
#         tufte.smoothed_time_series(ax,
#             data=data,
#             radius=1000, step=100
#         )
#     if False:
#         data = raw_data.T

# if True:
#     data = []
#     s = "name"
#     v = "_runtime"
#     for x in [
#         "ancient-serenity-60", # Tree
#         "brisk-dream-61", # Net
#         # "denim-haze-24", # None
        
#         # "stilted-fog-57", # Tree
#         # "dark-lake-59", # Net
#         # "prime-dream-66" # None
        
#         # "dulcet-planet-58",
#         # "summer-darkness-63",
#         # "dark-flower-64",
#         # "floral-jazz-65",
#     ]:

#         # data.append(pd.read_csv(f"results/pets_vs_sac/{s}={x}_{v}.csv")[499:-1:500].values)

#         # ROLLING CORRELATION
#         df = pd.read_csv(f"results/pets_vs_sac/name={x}_reward_sum_oracle.csv")
#         df["model"] = pd.read_csv(f"results/pets_vs_sac/name={x}_reward_sum_model.csv")
#         data.append(df[x].rolling(1000).corr(df["model"]).values[:50000:100])

#         # data.append(tufte.smoothed_time_series(ax,
#         #     csv_name=f"results/pets_vs_sac/{s}={x}_{v}", 
#         #     # radius=250, step=1000,
#         #     radius=1, step=1,
#         #     mean_mode=True,
#         #     q=[],#[10,20,30,40]
#         #     suppress_plot=True
#         # )[0])

# cbar_ax = tufte.coloured_2d_curves(ax, data)
# ax.show()
# ax.save("sac_corr.svg")
# # cbar_ax.save("cbar.svg")
"""
Rollout all final models, as well as oracle and random policies, saving out their trajectories as a dictionary of graphs.
Then evaluate the models by their regret reduction as a fraction of (oracle-access agent - random agent).
"""
import os
import argparse
from torch import save, load
from numpy import array, percentile
from scripts.evaluate.enjoy import load_and_deploy


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("--pets_version", type=int, default=3)
parser.add_argument("--dynamics_version", type=int, default=2)
parser.add_argument("--num_eps", type=int, default=100)
args = parser.parse_args()

save_path = f"final_model_rollout_graphs/fastjet/{args.task}/{args.oracle}"
if not os.path.exists(save_path): os.makedirs(save_path)
save_path += f"/{args.num_eps}e_0p.graphs"
if os.path.exists(save_path): graphs = load(save_path)
else:
    # Evaluate random and oracle as lower and upper bounds
    graphs = {}
    print("Random")
    graphs["random"] = load_and_deploy(args.task, args.oracle, None, args.pets_version, args.dynamics_version,
                                       args.num_eps, render_freq=0, explain_freq=0, random_agent=True)
    print("Oracle")
    graphs["oracle"] = load_and_deploy(args.task, args.oracle, None, args.pets_version, args.dynamics_version,
                                       args.num_eps, render_freq=0, explain_freq=0, random_agent=False)

    # Deploy an agent with each model and concatenate returns by subdirectory
    path = f"graphs_and_models/fastjet/{args.task}/{args.oracle}"
    for subdir in os.listdir(path):
        graphs[subdir] = {}
        for run in os.listdir(f"{path}/{subdir}"):
            name = f"{subdir}/{run}/0200"
            print(name)
            graphs[subdir][run] = load_and_deploy(args.task, args.oracle, name, args.pets_version, args.dynamics_version,
                                                  args.num_eps, render_freq=0, explain_freq=0, random_agent=False)
    save(graphs, save_path)

lo =  percentile(graphs["random"].oracle_returns, 50)
rng = percentile(graphs["oracle"].oracle_returns, 50) - lo

for k, v in graphs.items():
    if type(v) == dict:
        print(f"{k}:")
        all_regret_norm = []
        for kk, vv in v.items():
            regret_norm = (array(vv.oracle_returns) - lo) / rng
            all_regret_norm += list(regret_norm)
            print(f"  {kk}:".ljust(30), percentile(regret_norm, [25,50,75]))
        print(f"  ALL:".ljust(30), percentile(all_regret_norm, [25,50,75]))
    else:
        regret_norm = (array(v.oracle_returns) - lo) / rng
        print(f"{k}:".ljust(30), percentile(regret_norm, [25,50,75]))

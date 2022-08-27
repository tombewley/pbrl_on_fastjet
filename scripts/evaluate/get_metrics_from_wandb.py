import argparse
import yaml
from rlutils.experiments.wandb import get


parser = argparse.ArgumentParser()
parser.add_argument("username", type=str)
parser.add_argument("save_path", type=str)
parser.add_argument("--tag", type=str, default=None)
args = parser.parse_args()

# NOTE: Look for filters.yaml in the save path
with open(f"results/{args.save_path}/filters.yaml", "r") as f:
    filters = yaml.safe_load(f)

for data in get(
    project_name=f"{args.username}/pbrl_iclr",
    metrics=[
        "oracle_return"
    ],
    filters=filters,
    tag=args.tag
    ):
    for m in data:
        data[m]["df"].to_csv(f"results/{args.save_path}/{data[m]['fname']}", index=False)

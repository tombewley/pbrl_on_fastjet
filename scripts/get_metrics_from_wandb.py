import argparse
import yaml
from rlutils.experiments.wandb import get


parser = argparse.ArgumentParser()
parser.add_argument("project", type=str)
parser.add_argument("save_path", type=str)
parser.add_argument("--tag", type=str, default=None)
args = parser.parse_args()

# NOTE: Look for config.yaml in the save path
with open(f"results/{args.save_path}/config.yaml", "r") as f:
    config = yaml.safe_load(f)

for data in get(project_name=f"{args.project}", metrics=config["metrics"], filters=config["filters"], tag=args.tag):
    for m in data:
        data[m]["df"].to_csv(f"results/{args.save_path}/{data[m]['fname']}", index=False)

"""
Deploy PETS with a pretrained dynamics model and a learnt reward model.
"""

import argparse
import gym, fastjet
from torch import device, load
from torch.cuda import is_available
from rlutils import build_params, make, deploy
from rlutils.observers.pbrl import PbrlObserver


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("model", type=str)
parser.add_argument("--pets_version", type=int, default=2)
parser.add_argument("--dynamics_version", type=int, default=1)
parser.add_argument("--num_eps", type=int, default=100)
parser.add_argument("--render_freq", type=int, default=1)
parser.add_argument("--wandb", type=int, default=0)
args = parser.parse_args()

P = build_params(
    [f"agent.pets={args.pets_version}", f"env.fastjet.{args.task}", f"oracle.fastjet.{args.task}.{args.oracle}"],
    root_dir="config")

# Create environment
env = gym.make("FastJet-v0",
    task=args.task, 
    skip_frames=P["deployment"]["skip_frames"],
    render_mode="human" if args.render_freq > 0 else False,
    camera_angle=P["deployment"]["camera_angle"]
)

# Create PbrlObserver
P["pbrl"]["reward_source"] = "model"
pbrl = PbrlObserver(P["pbrl"])
device_ = device("cuda" if is_available() else "cpu")
pbrl.model = load(f"graphs_and_models/fastjet/{args.task}/{args.oracle}/{args.model}.reward", map_location=device_)

# Create agent
P["agent"]["pretrained_model"] = load(f"pretrained_dynamics/{args.task}_v{args.dynamics_version}.dynamics",
                                      map_location=device_)
P["agent"]["reward"] = pbrl.reward
agent = make("pets", env, hyperparameters=P["agent"])

# Deploy
deploy(agent=agent, P={
        "num_episodes": args.num_eps,
        "episode_time_limit": P["deployment"]["episode_time_limit"],
        "render_freq": args.render_freq,
    }
)

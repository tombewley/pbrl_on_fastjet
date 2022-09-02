"""
Use PETS with a pretrained dynamics model to generate an offline dataset
for fidelity evaluation.
"""

from ast import Or
import os
import argparse
import gym, fastjet
from torch import device, save, load
from torch.cuda import is_available
from random import randint
from rlutils import build_params, make, deploy
from rlutils.observers.pbrl import PbrlObserver
from rlutils.rewards.sampler import Sampler
from rlutils.rewards.interfaces import OracleInterface
from rlutils.rewards.interactions import preference_batch


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("oracle", type=str)
parser.add_argument("--dynamics_version", type=int, default=2)
parser.add_argument("--num_eps", type=int, default=200)
parser.add_argument("--num_preferences", type=int, default=1000)
parser.add_argument("--render_freq", type=int, default=0)
args = parser.parse_args()

# NOTE: PETS variant 2 as base parameters
P = build_params(
    ["agent.pets=2", f"env.fastjet.{args.task}", f"oracle.fastjet.{args.task}.{args.oracle}"],
    root_dir="config")

# Create environment
env = gym.make("FastJet-v0",
    task=args.task, 
    skip_frames=P["deployment"]["skip_frames"],
    render_mode="human" if args.render_freq > 0 else False,
    camera_angle=P["deployment"]["camera_angle"]
)

# Create PbrlObserver
P["pbrl"]["reward_source"] = "oracle"
P["pbrl"]["observe_freq"] = 1
pbrl = PbrlObserver(P["pbrl"])

# Create agent
P["agent"]["pretrained_model"] = load(f"pretrained_dynamics/{args.task}_v{args.dynamics_version}.dynamics",
                                      map_location=device("cuda" if is_available() else "cpu"))
P["agent"]["reward"] = pbrl.reward
agent = make("pets", env, hyperparameters=P["agent"])

# Define class for randomising the CEM parameters on each episode
class CEMUpdater:
    def __init__(self): self.agent, self.pbrl, self.run_names = agent, pbrl, []
    def per_timestep(*_): pass
    def per_episode(self, ep_num):
        if ep_num >= 0:
            # Store previous params
            self.pbrl.graph.nodes[ep_num]["cem_params"] = {k:v for k,v in agent.P.items() if "cem_" in k}
        # Update for next episode
        self.agent.P["cem_iterations"] = randint(1, 50)
        self.agent.P["cem_particles"] = randint(4, 50)
        self.agent.P["cem_elites"] = int(round(agent.P["cem_particles"] / 4))
        print(ep_num+1, {k:v for k,v in agent.P.items() if "cem_" in k})
        return {}
cem_updater = CEMUpdater()

# Deploy and record episodes
cem_updater.per_episode(-1)
deploy(agent=agent, P={
        "num_episodes": args.num_eps,
        "episode_time_limit": P["deployment"]["episode_time_limit"],
        "render_freq": args.render_freq,
        "observers": {"pbrl": pbrl, "cem_updater": cem_updater}
    }
)

# Collect a single preference batch at the end with uniform sampling.
sampler = Sampler(pbrl.graph, model=None, P={
        "weight": "uniform",
        "recency_constraint": False,
        "probabilistic": True
})
preference_batch(sampler, pbrl.interface, pbrl.graph, args.num_preferences)
print(pbrl.graph)

# Save out, ensuring no overwriting by incrementing index
save_dir = f"offline_graphs/fastjet/{args.task}/{args.oracle}"
save(pbrl.graph, f"{save_dir}/{len(os.listdir(save_dir))}_{len(pbrl.graph)}e_{len(pbrl.graph.edges)}p.graph")

"""
Use PETS with a pretrained dynamics model to generate an evaluation dataset
for Equivalent-Policy Invariant Comparison (EPIC).
"""

import gym, fastjet
from torch import device, save, load as load
from torch.cuda import is_available
from random import randint
from matplotlib.pyplot import show
from rlutils import build_params, make, deploy
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.pbrl.interfaces import OracleInterface
from config import oracles


TASK = "follow"
ORACLE = oracles.dist_closing_uperr_v2
DYNAMICS_PATH = "pretrained_dynamics/follow_v1.dynamics"

NUM_EPS = 100
RENDER = False


pbrl = PbrlObserver({
    "reward_source": "oracle",
    "observe_freq": 1,
    "interface": {
        "class": OracleInterface,
        "oracle": ORACLE
    }
})

P = build_params(["agent.pets"], root_dir="config")["agent"]
P["pretrained_model"] = load(DYNAMICS_PATH, map_location=device("cuda" if is_available() else "cpu"))
P["reward"] = pbrl.reward

agent = make("pets", hyperparameters=P,
    env=gym.make("FastJet-v0",
            task=TASK, 
            skip_frames=25,
            render_mode="human" if RENDER else False,
            camera_angle="outside_target_bg"
    )
)

class CEMUpdater:
    def __init__(self): self.agent, self.pbrl, self.run_names = agent, pbrl, []
    def per_timestep(*_): pass
    def per_episode(self, ep_num):
        # Randomise the CEM parameters for each episode
        self.agent.P["cem_iterations"] = randint(1, 50)
        self.agent.P["cem_particles"] = randint(4, 50)
        self.agent.P["cem_elites"] = int(round(agent.P["cem_particles"] / 4))
        self.pbrl.graph.nodes[ep_num]["cem_settings"] = {k:v for k,v in agent.P.items() if "cem_" in k}
        return {}

deploy(
    agent=agent,
    P={"num_episodes": NUM_EPS, "episode_time_limit": 20, "render_freq": int(RENDER)},
    observers={"pbrl": pbrl, "cem_updater": CEMUpdater()}
)
save(pbrl.graph, f"epic_datasets/{TASK}/{ORACLE.__name__}.graph")

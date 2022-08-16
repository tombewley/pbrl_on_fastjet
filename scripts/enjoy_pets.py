import gym, fastjet
from rlutils import build_params, make, deploy
from rlutils.observers.pbrl import load
from rlutils.observers.pets_explainer import PetsExplainer
from torch import device, load as pt_load
from torch.cuda import is_available
P = build_params(["agent.pets"], root_dir="config")["agent"]

"""
Use notebooks.deep_explanation?
"""


TASK = "follow"
DYNAMICS_PATH = "pretrained_dynamics/follow_v1.dynamics"
PBRL_PATH = "graphs_and_models/fastjet/follow/charmed-totem-113/200.pbrl"
RENDER = True
WANDB = False


pbrl = load(PBRL_PATH, {"reward_source": "model"})
P["reward"] = pbrl.reward
P["pretrained_model"] = pt_load(DYNAMICS_PATH, map_location=device("cuda" if is_available() else "cpu"))
agent = make("pets", hyperparameters=P,
    env=gym.make("FastJet-v0", 
            task=TASK, 
            skip_frames=25,
            render_mode="human" if RENDER else False,
            camera_angle="outside_target_bg"
    )
)

deploy(agent, {
        "num_episodes": 100, 
        "episode_time_limit": 20, 
        "render_freq": int(RENDER), 
        "do_extra": True,
        "wandb_monitor": WANDB,
    },
    observers={"pbrl": PetsExplainer(agent, reward_model=pbrl.model)},
    train=True
)

"""
Run PbRL
"""

import gym, fastjet, rlutils
from rlutils.observers.pbrl import PbrlObserver, OracleInterface
from rlutils.experiments.deploy import SumLogger
from config.features import F
from config.oracles import target_pose_linear, target_pose_tree

LOAD = None # "polar-smoke-105_ep500000"
TRAIN = True
SKIP_FRAMES = 25

# TODO: config.yaml

P={
    "project_name": "fastjet",
    "task": "target_no_reward",
    "agent": "sac",
    "num_episodes": 100000,
    "episode_time_limit": 500 / SKIP_FRAMES, # Num frames = episode_time_limit * skip_frames
    "skip_frames": SKIP_FRAMES,
    "wandb_monitor": True,
    "render_freq": 100,
    "checkpoint_freq": 0,
}
AP = {
    "sac_ian": {
        "net_pi": [(None, 512), "R", (512, 512), "R", (512, 256), "R", (256, None)],
        "net_Q": [(None, 512), "R", (512, 512), "R", (512, 256), "R", (256, None)],
        "replay_capacity": 1e6,
        "batch_size": 1024,
        "lr_pi": 5.0e-5,
        "lr_Q": 5.0e-5,
        "gamma": 0.95,
        "alpha": 0.2,
        "tau": 0.005,
        "update_freq": 50
    },  
    "sac": {
        "net_pi": [(None, 256), "R", (256, 256), "R", (256, None)],
        "net_Q": [(None, 256), "R", (256, 256), "R", (256, None)],
        "input_normaliser": "obs_lims",
        "replay_capacity": 5e5,
        "batch_size": 256, 
        "lr_pi": 1e-4, 
        "lr_Q": 1e-3,
        "gamma": 0.75, 
        "alpha": 0.2, 
        "tau": 0.005, 
        "update_freq": 15, 
    },
    "dqn": {
        "net_Q": [(None, 256), "R", (256, 128), "R", (128, 64), "R", (64, None)],
        "replay_capacity": 5e5,
        "batch_size": 256,
        "lr_Q": 1e-3,
        "epsilon_decay": P["num_episodes"]*P["episode_time_limit"],
    },
    "steve": {
        "net_model": [(None, 32), "R", (32, 64), "R", (64, None)],
        "replay_capacity": 1e6,
        "num_random_steps": 0, 
        "num_models": 2, 
        "model_freq": 1, 
        "lr_model": 1e-3, 
        "horizon": 5, 
        "ddpg_parameters": {"td3": True},
        "nonfixed_dim": 19 # NOTE: First 19 dimensions vary over an episode
    }
}
PP = {
        # "interface": (fastjet.Interface,), 
        "interface": (OracleInterface, target_pose_tree), 

        "reward_source": "oracle",

        # Comment out the below if reward_source = oracle
        # "feedback_budget": 10000,
        # "observe_freq": 50, 
        # "feedback_freq": 500, # By ep not n
        # "num_episodes_before_freeze": 50000, 
        # "scheduling_coef": 0,
        # "sampling": {
        #     "weight": "ucb", 
        #     "constrained": True, 
        #     "probabilistic": True, # NOTE: Very early indication that this may be better.
        #     "num_std": 2
        #     },
        # "p_clip": 0.1,
        # "m_max": 100,
        # "m_stop_merge": 1, 
        # "min_samples_leaf": 1,
        # "alpha": 0.005,
        # "save_freq": 1000000, # By ep not n
        # "log_freq": 500, # By ep not n        
    }
if P["agent"] == "dqn": 
    PP["discrete_action_map"] = fastjet.env.DISCRETE_ACTION_MAP
if P["agent"] == "steve": 
    assert not P["normalise"]

env = gym.make("FastJet-v0", 
    task=P["task"], 
    continuous=(P["agent"] != "dqn"), 
    skip_frames=P["skip_frames"],
    render_mode=("human" if P["render_freq"] > 0 or TRAIN is False else False),
    camera_angle="target"
)

if LOAD:
    agent = rlutils.load(f"agents/{LOAD}.agent", env)
    agent.start()
else:
    agent = rlutils.make(P["agent"], env=env, hyperparameters=AP[P["agent"]])

pbrl = PbrlObserver(P=PP, features=F)
# pbrl.link(agent) assigns intrinsic reward function.
if PP["reward_source"] != "extrinsic": pbrl.link(agent)    

_, run_name = rlutils.deploy(
    agent,
    P=P if TRAIN else {
        "num_episodes": int(1e6), 
        "episode_time_limit": P["episode_time_limit"], 
        "render_freq": 1
        },
    train=TRAIN,
    observers={
        "pbrl": pbrl,
        "phase_counter": SumLogger({
            "name": "phase", 
            "source": "info", 
            "key": "phase"
        })
    }
)
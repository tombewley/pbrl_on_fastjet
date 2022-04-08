"""
Simplified script for running preference-based reinforcement learning,
with exemplar parameters and reduced use of functions from rlutils.
"""

import fastjet
from gym import make as make_env, wrappers
from rlutils import make as make_agent
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.pbrl.models import RewardTree
from rlutils.observers.pbrl.interfaces import OracleInterface

from torch import from_numpy

from config.features import preprocessor, features
from config.oracles import dist_closing_uperr
from config.interface import FastJetInterface
from config.params.agent.sac import P as P_sac
from config.params.agent.pets import P as P_pets


ORACLE_MODE = True # Whether to use synthetic oracle preferences (False = human preferences via web app)
NUM_EPISODES = 100000
AGENT_SAVE_FREQ = 10000
TIME_LIMIT = 20 # Limit on episode length
RENDER_ON = False
WANDB_ON = False # Whether to enable Weights & Biases monitoring


env = wrappers.TimeLimit(make_env("FastJet-v0",
    task="follow", # Task variant; specified in fastjet.tasks
    continuous=True, # Continuous or discrete actions
    skip_frames=25, # Number of simulator frames to run per timestep
    render_mode=("human" if RENDER_ON else False),
    camera_angle="outside_target_bg"
    ),
    max_episode_steps=TIME_LIMIT)

# PbrlObserver is the master class that handles the reward learning process.
pbrl = PbrlObserver(
    P={
        "feedback_budget": 10000, # Total number of preferences to be collected
        "observe_freq": 50, # Proportion of episodes stored for use in preference learning (higher = smaller proportion)
        "feedback_freq": 100, # Frequency of preference batches/updates
        "num_episodes_before_freeze": 50000, # Period over which preferences are collected
        "scheduling_coef": 0, # Whether to bias preference collection to later in the period (deprecated)
        "reward_source": "model",
        # A set of features to use in the reward function, derived from (s,a,s') tuples
        "featuriser": {
            "preprocessor": preprocessor,
            "features": features
        },
        "model": {
            "class": RewardTree, # Use tree as the reward function model
            "preference_eqn": "thurstone", # Use Thurstone's preference equation
            "loss_func": "bce", # Use binary cross-entropy loss function
            "split_by_variance": True, # Use variance-based splitting method
            # Additional tree growth parameters...
            "p_clip": 0.1, "m_max": 100, "num_from_queue": float("inf"), "min_samples_leaf": 1, "store_all_qual": False, "alpha": 0.001,
        },
        # Parameters for trajectory pair sampling strategy
        "sampler": {"weight": "ucb", "constrained": True, "probabilistic": True, "num_std": 0},
        "interface": {
            "class": OracleInterface, # Use synthetic oracle preferences
            "oracle": dist_closing_uperr # Ground-truth reward function to use in OracleInterface
            
        } if ORACLE_MODE else {
            "class": FastJetInterface, # Use human preferences via VPython interface
        }
    }
)

# === AGENT-SPECIFIC CREATION ====
# Give the agent access to the learnt reward function.
# NOTE: This is the trickiest functionality to make agent-agnostic.
if False: 
    """
    For off-policy model-free algorithms (e.g. SAC), need to link to the replay memory. This enables labelling
    of transitions with rewards, and relabelling of the entire memory when the reward function is updated.
    Requires:
      (1) The agent class to have access to pbrl.reward for labelling each new (s,a,s').
      (2) The pbrl class to have a "callback" method (pbrl.relabel_memory) which triggers
          the relabelling process after each update.
    """
    agent = make_agent("sac", env=env, hyperparameters=P_sac["agent"])
    agent.memory.__init__(agent.memory.capacity, reward=pbrl.reward, relabel_mode="eager")
    pbrl.relabel_memory = agent.memory.relabel
if True:
    """
    For the model-based algorithms with decision-time planning (e.g. PETS), need to give the model
    access to the reward function.
    """
    P_pets["agent"]["reward"] = pbrl.reward
    agent = make_agent("pets", env=env, hyperparameters=P_pets["agent"])

# ================================

if WANDB_ON:
    import wandb, os
    run = wandb.init(project="fastjet-follow")
    save_dir = f"agents/{run.name}"
    os.makedirs(save_dir, exist_ok=True)

for ep in range(NUM_EPISODES):
    state, done = env.reset(), False
    while not done:

        # === AGENT-SPECIFIC ACTION SELECTION ===
        state_torch = from_numpy(state).float().to(agent.device).unsqueeze(0)
        action, _ = agent.act(state_torch)
        # =======================================

        # NOTE: "extrinsic" reward from env is always zero. While here it's getting
        # passed into the agent.per_timestep method, it never gets used. Instead,
        # rewards in the replay memory are computed by pbrl.reward, and overwritten
        # whenever pbrl.relabel_memory is called.
        next_state, reward, done, _ = env.step(action)

        # === AGENT-SPECIFIC UPDATE ===
        next_state_torch = from_numpy(next_state).float().to(agent.device).unsqueeze(0)
        agent.per_timestep(state_torch, action, reward, next_state_torch, done)
        # =============================

        # Store the latest transition (s,a,s') in the PbRL class. 
        # NOTE: pbrl.per_timestep has nine arguments to make it behave like other
        # classes in the rlutils library, but only three currently get used.
        pbrl.per_timestep(None, None, state, action, next_state, None, None, None, None)

        state, state_torch = next_state, next_state_torch

    # Once-per episode method which stores or discards the latest episode according to
    # observe_freq, outputs some logs, and triggers a preference batch/reward update 
    # according to feedback_freq. The next episode does not begin until the update is complete.
    logs = pbrl.per_episode(ep)
    print(ep, "\n", logs)
    if WANDB_ON: wandb.log(logs)

    if (ep+1) % AGENT_SAVE_FREQ == 0: agent.save(f"{save_dir}/{ep+1}")

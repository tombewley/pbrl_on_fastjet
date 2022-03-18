"""
Script for running preference-based reinforcement learning.
"""

import sys
import importlib
from pprint import pprint
import torch
torch.set_printoptions(precision=3, linewidth=100000)   
import gym

import fastjet
import rlutils
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.sum_logger import SumLogger

from config.features import F
from config.params.base import P


def recursive_update(d1, d2, i=None, block_overwrite=False, verbose=False):
    # Adapted from https://stackoverflow.com/a/38504949.
    # TODO: Wheel reinvention here! Use wandb.config
    def _update(d1, d2, path=[]):
        for k in d2:
            typ = None
            if k in d1:
                if isinstance(d1[k], dict) and isinstance(d2[k], dict): _update(d1[k], d2[k], path+[k])
                elif block_overwrite: raise Exception(f"{'.'.join(path+[k])}: {d1[k]} | {d2[k]}")
                else: typ = "UP "
            else: typ = "NEW"
            if typ is not None: 
                d1[k] = d2[k]
                if verbose: print(f"{typ} {'.'.join(path+[k])}: {d1[k]}")
    def _select_index(d1):
        for k in d1: 
            if isinstance(d1[k], dict): _select_index(d1[k])
            elif isinstance(d1[k], list): d1[k] = d1[k][i]
    _update(d1, d2)
    if i is not None: _select_index(d1)

if __name__ == "__main__":
    P_update = {}
    for p in sys.argv[1:]:
        i = None
        try:
            if "=" in p: # For parameter array
                p, i = p.split("="); i = int(i) 
            P_new = importlib.import_module(f"config.params.{p}").P
        except ImportError: # If not a recognised config file, treat as filename for loading
            P_new = {"deployment": {"agent_load_fname": p}} 
        recursive_update(P_update, 
            P_new,
            i=i,
            block_overwrite=True,
            verbose=False
            )
    recursive_update(P, P_update, verbose=True)
    P["deployment"]["project_name"] = "fastjet-" + P["deployment"]["task"]

    pprint(P)

    # Sense checks
    if P["deployment"]["agent"] == "dqn": 
        P["pbrl"]["discrete_action_map"] = fastjet.env.DISCRETE_ACTION_MAP
    if P["deployment"]["agent"] in {"steve", "simple_model_based"}: 
        assert "input_normaliser" not in P["agent"]

    env = gym.make("FastJet-v0", 
        task=P["deployment"]["task"], 
        continuous=(P["deployment"]["agent"] != "dqn"), 
        skip_frames=P["deployment"]["skip_frames"],
        render_mode=("human" if "render_freq" in P["deployment"] 
                     and P["deployment"]["render_freq"] > 0 else False),
        camera_angle="outside_target_bg"
    )

    agent_type = P["deployment"]["agent"]
    agent_params = P["agent"][P["deployment"]["agent"]]

    pbrl = PbrlObserver(P=P["pbrl"], features=F)
    do_link = False
    if P["deployment"]["train"] and P["pbrl"]["reward_source"] != "extrinsic": 
        if P["deployment"]["agent"] in {"steve", "simple_model_based"}: 
            agent_params["reward"] = pbrl.reward
        else: do_link = True

    if "agent_load_fname" in P["deployment"]:
        fname = P["deployment"]["agent_load_fname"]
        agent = rlutils.load(f"agents/{fname}.agent", env)
        if P["deployment"]["train"]: agent.start()
        print(f"Loaded {fname}")
    else:
        agent = rlutils.make(agent_type, env=env, hyperparameters=agent_params)
    if do_link: pbrl.link(agent)

    rlutils.deploy(agent, P=P["deployment"], train=P["deployment"]["train"],
        observers={
            "pbrl": pbrl,
            "phase_counter": SumLogger({
                "name": "phase", 
                "source": "info", 
                "key": "phase"
            })
        }
    )
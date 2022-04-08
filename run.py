"""
Script for running preference-based reinforcement learning.
"""

import sys
from pprint import pprint
import torch
torch.set_printoptions(precision=3, linewidth=100000)   
import gym

import fastjet
import rlutils
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.sum_logger import SumLogger

from config.params.base import P as base


if __name__ == "__main__":
    P = rlutils.build_params(sys.argv[1:], base, root_dir="config.params")
    P["deployment"]["project_name"] = "fastjet-" + P["deployment"]["task"]

    pprint(P)

    if P["deployment"]["agent"] == "dqn": 
        P["pbrl"]["discrete_action_map"] = fastjet.env.DISCRETE_ACTION_MAP

    env = gym.make("FastJet-v0", 
        task=P["deployment"]["task"], 
        continuous=(P["deployment"]["agent"] != "dqn"), 
        skip_frames=P["deployment"]["skip_frames"],
        render_mode=("human" if "render_freq" in P["deployment"] 
                     and P["deployment"]["render_freq"] > 0 else False),
        camera_angle="outside_target_bg"
    )

    pbrl = PbrlObserver(P=P["pbrl"])

    do_link = False
    if "agent_load_fname" in P["deployment"]:
        fname = P["deployment"]["agent_load_fname"]
        agent = rlutils.load(f"agents/{fname}.agent", env)        
        # if P["deployment"]["train"]: agent.start()
        print(f"Loaded {fname}")
    else:
        if P["pbrl"]["reward_source"] != "extrinsic":
            if P["deployment"]["agent"] in {"steve", "pets", "mbpo"}:
                P["agent"]["reward"] = pbrl.reward
                if P["deployment"]["agent"] == "mbpo":
                    raise Exception("Link to MBPO rollouts not memory!")
            else:
                do_link = True

        agent = rlutils.make(P["deployment"]["agent"], env=env, hyperparameters=P["agent"])

    if do_link: pbrl.link(agent)

    # pprint(agent.P)

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

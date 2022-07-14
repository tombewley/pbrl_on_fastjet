"""
Script for running preference-based reinforcement learning.
"""

import sys
from pprint import pprint
import gym

import rlutils
import fastjet
import holonav
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.sum_logger import SumLogger
from rlutils.common.env_wrappers import DoneWipeWrapper

from config.base import P as base


if __name__ == "__main__":
    P = rlutils.build_params(sys.argv[1:], base, root_dir="config")
    pprint(P)

    if P["deployment"]["env"] == "FastJet-v0":
        env = gym.make(P["deployment"]["env"],
            task=P["deployment"]["task"],
            continuous=(P["deployment"]["agent"] != "dqn"),
            skip_frames=P["deployment"]["skip_frames"],
            render_mode=("human" if "render_freq" in P["deployment"]
                        and P["deployment"]["render_freq"] > 0 else False),
            camera_angle="outside_target_bg"
        )
    else:
        env = DoneWipeWrapper(gym.make(P["deployment"]["env"]))

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

    rlutils.deploy(agent, P=P["deployment"], train=P["deployment"]["train"],
        observers={
            "pbrl": pbrl,
            # "phase_counter": SumLogger({
            #     "name": "phase",
            #     "source": "info",
            #     "key": "phase"
            # })
        }
    )

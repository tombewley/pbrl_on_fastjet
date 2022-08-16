"""
Script for running preference-based reinforcement learning.
"""

import sys
from pprint import pprint
import gym
from torch import device, load
from torch.cuda import is_available

import rlutils
import fastjet
import holonav
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.loggers import SumLogger
from rlutils.common.env_wrappers import DoneWipeWrapper

from config.base import P as base


if __name__ == "__main__":
    P = rlutils.build_params(sys.argv[1:], base, root_dir="config")
    pprint(P)

    if P["deployment"]["env"] == "FastJet-v0":
        env = gym.make("FastJet-v0",
            task=P["deployment"]["task"],
            continuous=(P["deployment"]["agent"] != "dqn"),
            skip_frames=P["deployment"]["skip_frames"],
            render_mode=("human" if "render_freq" in P["deployment"]
                        and P["deployment"]["render_freq"] > 0 else False),
            camera_angle=P["deployment"]["camera_angle"]
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
                if P["agent"]["pretrained_model"]:
                    # NOTE: Loading pretrained model here
                    P["agent"]["pretrained_model"] = load(
                        f"pretrained_dynamics/{P['deployment']['task']}_v1.dynamics",
                        map_location=device("cuda" if is_available() else "cpu"))
                if P["deployment"]["agent"] == "mbpo":
                    raise Exception("Link to MBPO rollouts not memory!")
            else:
                do_link = True

        agent = rlutils.make(P["deployment"]["agent"], env=env, hyperparameters=P["agent"])

    if do_link: pbrl.link(agent)

    P["deployment"]["observers"]["pbrl"] = pbrl
    rlutils.deploy(agent, P=P["deployment"], train=P["deployment"]["train"])

"""
Script for running preference-based reinforcement learning.
"""

import argparse
from pprint import pprint
import gym
from torch import device, load
from torch.cuda import is_available
import wandb

import rlutils
import fastjet
import holonav
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.loggers import SumLogger
from rlutils.common.env_wrappers import DoneWipeWrapper

from config.base import P as base


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("env", type=str)
    parser.add_argument("agent", type=str)
    parser.add_argument("num_eps", type=int)

    parser.add_argument("--oracle", type=str) # Will prepend env
    parser.add_argument("--human", type=int, default=0)
    parser.add_argument("--model", type=str)
    parser.add_argument("--features", type=str) # Will prepend env
    parser.add_argument("--schedule", type=str)
    parser.add_argument("--sampler", type=str)

    parser.add_argument("--render_freq", type=int, default=0)
    parser.add_argument("--save_freq", type=int, default=0)
    parser.add_argument("--wandb", type=int, default=0)
    parser.add_argument("--group", type=str)
    parser.add_argument("--job_type", type=str)

    args = parser.parse_args()

    if args.wandb:
        assert args.group is not None, "Need to specify group if using Weights & Biases"
        wandb_config = {
            "project": "pbrl_iclr",
            "group": args.group,
            "job_type": args.job_type
        }
    else: wandb_config = None

    # Build configs for various components of the system by reading in parameter dictionaries
    P = rlutils.build_params([
        f"env.{args.env}",
        f"agent.{args.agent}",
        f"oracle.{args.env}.{args.oracle}" if args.oracle is not None else "",
        f"interface.{'fastjet' if 'fastjet' in args.env else 'video'}" if args.human else "",
        f"model.{args.model}" if args.model is not None else "",
        f"features.{args.env}.{args.features}" if args.features is not None else "",
        f"schedule.{args.schedule}" if args.schedule is not None else "",
        f"sampler.{args.sampler}" if args.sampler is not None else ""
        ], base, root_dir="config")
    pprint(P)

    if P["deployment"]["env"] == "FastJet-v0":
        env = gym.make("FastJet-v0",
            task=P["deployment"]["task"],
            continuous=(P["deployment"]["agent"] != "dqn"),
            skip_frames=P["deployment"]["skip_frames"],
            render_mode=("human" if args.render_freq > 0 else False),
            camera_angle=P["deployment"]["camera_angle"]
        )
    else:
        env = DoneWipeWrapper(gym.make(P["deployment"]["env"]))

    P["pbrl"]["save_freq"] = args.save_freq
    pbrl = PbrlObserver(P=P["pbrl"])

    do_link = False
    if "agent_load_fname" in P["deployment"]:
        fname = P["deployment"]["agent_load_fname"]
        agent = rlutils.load(f"agents/{fname}.agent", env)        
        # if P["deployment"]["train"]: agent.start()
        print(f"Loaded {fname}")
    else:
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

    P["deployment"]["num_episodes"] = args.num_eps
    P["deployment"]["observers"]["pbrl"] = pbrl
    P["deployment"]["render_freq"] = args.render_freq
    rlutils.deploy(agent, P=P["deployment"], train=P["deployment"]["train"], wandb_config=wandb_config)

"""
Deploy PETS with a pretrained dynamics model and a learnt reward model.
"""

import argparse
from torch import device, load
from torch.cuda import is_available
import gym, fastjet
from rlutils import build_params, make, deploy
from rlutils.observers.pbrl import PbrlObserver


def load_and_deploy(task, oracle, model, pets_version, dynamics_version,
                         num_eps, render_freq, explain_freq, random_agent):
    P = build_params(
        [f"agent.pets={pets_version}", f"env.fastjet.{task}", f"oracle.fastjet.{task}.{oracle}"],
        root_dir="config")
    P["pbrl"]["observe_freq"] = 1 # NOTE: For evaluation
    P["pbrl"]["explainer"] = {
        "save": False,
        "show": True,
        "freq": explain_freq,
        "plots": [
            "leaf_visitation_heatmap",
            "leaf_visitation_time_series",
        ]
    } if explain_freq else {}

    # Create environment
    env = gym.make("FastJet-v0",
        task=task, 
        skip_frames=P["deployment"]["skip_frames"],
        render_mode="human" if render_freq > 0 else False,
        camera_angle=P["deployment"]["camera_angle"]
    )

    # Create PbrlObserver
    pbrl = PbrlObserver(P["pbrl"])
    device_ = device("cuda" if is_available() else "cpu")
    if model is None:
        P["pbrl"]["reward_source"] = "oracle"
    else:
        P["pbrl"]["reward_source"] = "model"
        pbrl.model = load(f"graphs_and_models/fastjet/{task}/{oracle}/{model}.reward", map_location=device_)
        pbrl.model.device = device_

    # Create agent
    if random_agent:
        agent = make("random", env)
    else:
        P["agent"]["pretrained_model"] = load(f"pretrained_dynamics/{task}_v{dynamics_version}.dynamics",
                                              map_location=device_)
        P["agent"]["reward"] = pbrl.reward
        agent = make("pets", env, hyperparameters=P["agent"])

    # Deploy
    deploy(agent=agent, P={
            "num_episodes": num_eps,
            "episode_time_limit": P["deployment"]["episode_time_limit"],
            "render_freq": render_freq,
            "observers": {"pbrl": pbrl}
        }
    )
    return pbrl.graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str)
    parser.add_argument("oracle", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--pets_version", type=int, default=2)
    parser.add_argument("--dynamics_version", type=int, default=2)
    parser.add_argument("--num_eps", type=int, default=100)
    parser.add_argument("--render_freq", type=int, default=1)
    parser.add_argument("--explain_freq", type=int, default=1)
    args = parser.parse_args()
    load_and_deploy(args.task, args.oracle, args.model, args.pets_version, args.dynamics_version,
                    args.num_eps, args.render_freq, args.explain_freq, random_agent=False)

"""
Pretrain a dynamics model on a random policy.
"""
import argparse
import gym, fastjet
from rlutils import build_params, make, train
from numpy import mean
from pprint import pprint


parser = argparse.ArgumentParser()
parser.add_argument("task", type=str)
parser.add_argument("dynamics_version", type=int)
parser.add_argument("--num_steps", type=int, default=int(1e5))
parser.add_argument("--num_updates", type=int, default=int(1e5))
args = parser.parse_args()

P = build_params([f"env.fastjet.{args.task}", "agent.pets_pretrain"], root_dir="config")
P["deployment"]["num_steps"] = P["agent"]["num_random_steps"] = args.num_steps


env = gym.make("FastJet-v0", task=P["deployment"]["task"], continuous=True, skip_frames=P["deployment"]["skip_frames"])
agent = make(P["deployment"]["agent"], env=env, hyperparameters=P["agent"])

pprint(P["deployment"])
pprint(agent.P)

train(agent, P=P["deployment"])

print(len(agent.random_memory))

for i in range(int(args.num_updates)):
    agent.update_on_batch()
    print(i, mean(agent.ep_losses))
    del agent.ep_losses[:]

agent.save(f"pretrained_dynamics/{args.task}_v{args.dynamics_version}")

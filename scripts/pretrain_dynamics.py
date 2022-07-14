"""
Pretrain a dynamics model on a random policy.
"""
import gym, fastjet
from rlutils import build_params, make, train
from numpy import mean
from torch import save
from pprint import pprint


TASK = "chase"
NUM_STEPS = int(1e5)
NUM_UPDATES = int(1e5)

P = build_params([f"env.fastjet.{TASK}", "agent.pets_pretrain"], root_dir="config")
P["deployment"]["num_steps"] = P["agent"]["num_random_steps"] = NUM_STEPS


env = gym.make("FastJet-v0", task=P["deployment"]["task"], continuous=True, skip_frames=P["deployment"]["skip_frames"])
agent = make(P["deployment"]["agent"], env=env, hyperparameters=P["agent"])

pprint(P["deployment"])
pprint(agent.P)

train(agent, P=P["deployment"])

print(len(agent.random_memory))

for i in range(int(NUM_UPDATES)):
    agent.update_on_batch()
    print(i, mean(agent.ep_losses))
    del agent.ep_losses[:]

agent.save(f"pretrained_dynamics/{TASK}_v1")

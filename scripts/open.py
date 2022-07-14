from config.oracle.pendulum.default import oracle
from rlutils.observers.pbrl import load

pbrl = load("models/pendulum/dainty-thunder-5/50.pbrl", {})

print(len(pbrl.graph))

rewards = [oracle(ep["transitions"]) for _, ep in pbrl.graph.nodes(data=True)]

print(rewards[0])
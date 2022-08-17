import gym, fastjet
from rlutils import make, deploy
from rlutils.observers.pbrl import PbrlObserver
from rlutils.common.featuriser import Featuriser
import matplotlib.pyplot as plt
from config.features.fastjet import *

f = Featuriser({"features": [
    los_error
]})

env = gym.make("FastJet-v0", task="chase", skip_frames=3, render_mode="human", camera_angle="outside_target_bg_offset")
pbrl = PbrlObserver({"observe_freq": 1})
deploy(make("random", env), {"num_episodes": 1, "episode_time_limit": 100, "render_freq": 1, "observers": {"pbrl": pbrl}})
features = f(pbrl.graph.states[0], pbrl.graph.actions[0], pbrl.graph.next_states[0])

_, axes = plt.subplots(1, features.shape[1], squeeze=False); axes = axes.flatten()
for i, (ax, ft) in enumerate(zip(axes, features.T)):
    ax.plot(ft)
    ax.set_title(f.names[i])
plt.show()

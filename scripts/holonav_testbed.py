import gym
import torch
import matplotlib.pyplot as plt
import holonav
from rlutils import make, deploy
from rlutils.observers.pbrl import PbrlObserver
from rlutils.observers.pbrl.interfaces import OracleInterface
from rlutils.observers.pbrl.models import RewardNet, RewardTree
from rlutils.observers.pbrl.interactions import preference_batch

DP = {
    "num_episodes": 50, 
    "render_freq": 0, 
    "episode_time_limit": 100
}
PP = {
    "observe_freq": 1,
    "sampler": {
        "weight": "uniform",
        "recency_constraint": False,
        "probabilistic": True
    },
    "interface": {
        "class": OracleInterface, 
        "oracle": lambda tr: (
            # tr[:,0] > 0.5
            # tr[:,2] > 0
            torch.logical_and(tr[:,1] >= .25, tr[:,1] < .75)
            # tr[:,0] + (tr[:,1] - 0.5)**2
            # tr[:,0] + (tr[:,1] - 0.5)**2 - 40*tr[:,2] + 40*tr[:,3]
            ).cpu().numpy()
    },
    "reward_source": "model",
    "model": {
        "featuriser": {
            "feature_names": ["x","y","vx","vy","x'","y'"]
        },
        "class": RewardTree,
        "preference_eqn": "thurstone",
        "loss_func": "0-1",
        "split_by_variance": True,
        "p_clip": 0.1,
        "m_max": 100,
        "num_from_queue": float("inf"),
        "min_samples_leaf": 1,
        "alpha": 0,
        "store_all_qual": True,
        
        # "class": RewardNet,
        # "preference_eqn": "bradley-terry",
        # "num_batches_per_update": 100,
        # "batch_size": 32,
    }
}

SEED = 0

env = gym.make("HoloNav-v0", 
    render_mode="human" if DP["render_freq"] > 0 else False,
    map={"shape": [1,1], "max_speed": .025,
        "boxes": {"init": {"coords": [[0,0],[1,1]], "init_weight": 1}}} 
)
agent = make("random", env)
pbrl = PbrlObserver(P=PP)

env.seed(SEED); pbrl.sampler.seed(SEED); pbrl.graph.seed(SEED)

deploy(agent, P=DP, observers={"pbrl": pbrl})
preference_batch(
    sampler=pbrl.sampler,
    interface=pbrl.interface,
    graph=pbrl.graph,
    batch_size=100,
    ij_min=0,
    history_key=0,
    budget=1
)

pbrl.model.update(graph=pbrl.graph, history_key=0)

pbrl.explainer.plot_loss_vs_m(0)

pbrl.model.show_split_quality(pbrl.model.tree.root)
# pbrl.model.tree.root.all_qual = pbrl.model.tree.root._proxy_qual
# pbrl.model.show_split_quality(pbrl.model.tree.root)

# pbrl.model.show_split_quality(pbrl.model.tree.root.left)
# pbrl.model.tree.root.left.all_qual = pbrl.model.tree.root.left._proxy_qual
# pbrl.model.show_split_quality(pbrl.model.tree.root.left)

# pbrl.model.show_split_quality(pbrl.model.tree.root.right)
# pbrl.model.tree.root.right.all_qual = pbrl.model.tree.root.right._proxy_qual
# pbrl.model.show_split_quality(pbrl.model.tree.root.right)

plt.ioff(); plt.show()
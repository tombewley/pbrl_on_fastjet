from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *

def oracle(s, a, ns):
    d = dist(s, a, ns, None)
    return - (0.1*d + 0.1*g_force(s, a, ns, None) + (d < 50).float() * (fwd_error(s, a, ns, None) + up_error(s, a, ns, None)))

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/match/dist_pose_when_close",
        "offline_graph_path": "offline_graphs/fastjet/match/dist_pose_when_close/0_200e_1000p.graph"
    }
}

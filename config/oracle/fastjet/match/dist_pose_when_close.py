from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *

def oracle(s, a, ns):
    d = dist(s, a, ns, None)
    return - (d + (d < 30).float() * (fwd_error(s, a, ns, None) + up_error(s, a, ns, None)))
    d = dist(t, None); close = (d < CONFIG["radius"]).float()
    return - (d + 10. * close * (fwd_error(t, None) + up_error(t, None)))

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/match/dist_pose_when_close",
        "offline_graph_path": "offline_graphs/fastjet/match/dist_pose_when_close/___.graph"
    }
}

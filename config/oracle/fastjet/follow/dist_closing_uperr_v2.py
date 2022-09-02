from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *

def oracle(s, a, ns):
    d = dist(s, a, ns, None)
    return - (d + 0.05 * closing_speed(s, a, ns, {"dist": d}) + 10. * up_error(s, a, ns, None))

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/follow/dist_closing_uperr_v2",
        "offline_graph_path": "offline_graphs/fastjet/follow/dist_closing_uperr_v2/1_200e_1000p.graph"
    }
}

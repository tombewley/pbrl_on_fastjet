from torch import abs
from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *

def oracle(s, a, ns):
    s, a, ns = preprocessor(s, a, ns) # NOTE: required to normalise attitude vectors
    return - (
          abs(dist(s, a, ns, None) -20.) \
        + 10. * los_error(s, a, ns, None) \
        + 5.0 * abs(roll(s, a, ns, None)) \
        + 100. * (alt(s, a, ns, None) < 50))

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/chase/dist20_los_roll_alt100",
        "offline_graph_path": "offline_graphs/fastjet/chase/dist20_los_roll_alt100/0_100e_4950p.graph"
    }
}

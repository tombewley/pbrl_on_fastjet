from torch import pi, acos, clamp
from rlutils.rewards.interfaces import OracleInterface

MAX_TORQUE = 2.

def oracle(s, a, ns):
    def angle_normalize(th): return ((th + pi) % (2 * pi)) - pi
    return - (angle_normalize(acos(clamp(s[...,0], -1, 1))) ** 2 + \
            .1 * s[...,2] ** 2 + \
            .001 * (clamp(a[...,0], -MAX_TORQUE, MAX_TORQUE) ** 2))

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/pendulum/default",
        # "offline_graph_path": "offline_graphs/pendulum/default/___.graph"
    }
}

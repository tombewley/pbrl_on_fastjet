from torch import abs, clamp, pi
from torch.linalg import norm
from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *

def oracle(s, a, ns):
    s, a, ns = preprocessor(s, a, ns) # NOTE: required to normalise attitude vectors

    not_closing_xz = (delta_dist_xz(s, a, ns, {"dist_xz": dist_xz(s, a, ns, None)}) > 0.).float()
    # NOTE: This just replicates delta_alt_error
    not_closing_y =  (abs(ns[...,1] - ns[...,20]) - abs(s[...,1] - s[...,20]) > 0.).float()
    abs_lr_offset_ = abs(lr_offset(s, a, ns, None))
    off_line =       (abs_lr_offset_ > 10.).float()
    thrust_ = thrust(s, a, ns, None)

    return - (
           1.    * not_closing_xz + \
           2.    * not_closing_y + \
           1.    * off_line + \

           0.05  * abs_lr_offset_ + \
           0.05  * alt(s, a, ns, None) + \

           1.    * hdg_error(s, a, ns, None) + \
           0.5   * pitch_error(s, a, ns, None) + \
           1.    * abs_roll(s, a, ns, None) + \

           0.25  * yaw_rate(s, a, ns, None) + \
           0.25  * roll_rate(s, a, ns, None) + \
           0.25  * pitch_rate(s, a, ns, None) + \

           0.1   * g_force(s, a, ns, None) + \
           0.025 * thrust_ + \
           0.05  * delta_thrust(s, a, ns, {"thrust": thrust_}) + \

           10.   * (alt(s, a, ns, None) < 0.5)
    )

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/land/closing_and_shaping",
        "offline_graph_path": "offline_graphs/fastjet/land/closing_and_shaping/0_200e_1000p.graph"
    }
}

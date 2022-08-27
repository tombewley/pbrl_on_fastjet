from torch import abs, clamp, pi
from torch.linalg import norm
from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *
from ....features.fastjet import _vec_to_target, _target_right # Private

PITCH_THRESHOLD = 0.3

def oracle(s, a, ns):


    v = _vec_to_target(ns)
    offset = cosim(_target_right(ns), v) * norm(v, axis=-1) # TODO: Make a feature


    s, a, ns = preprocessor(s, a, ns) # NOTE: required to normalise attitude vectors
    is_closing_xz = (delta_dist_xz(s, a, ns, {"dist_xz": dist_xz(s, a, ns, None)}) < 0.).float()
    is_closing_y = (abs(ns[...,1] - ns[...,20]) - abs(s[...,1] - s[...,20]) < 0.).float() # TODO: Make a feature
    on_line = (abs(offset) < 10.).float()

    pitch_error_ = 1.    * clamp(0.5 - abs(pitch(s, a, ns, None) - target_pitch(s, a, ns, None)), 0.)
    abs_roll_ =    1.    * clamp(pi/2 - abs(roll(s, a, ns, None)), 0.)
    hdg_error_ =   1.    * clamp(0.5 - hdg_error(s, a, ns, None), 0.)
    alt_ =         0.025 * clamp(20. - abs(alt(s, a, ns, None)), 0.)
    offset_ =      0.05  * clamp(60. - abs(offset), 0.)
    g_force_ =     0.1   * clamp(4. - g_force(s, a, ns, None), 0.)

    return is_closing_xz + is_closing_y + on_line + \
            pitch_error_ + \
            abs_roll_ + \
            hdg_error_ + \
            alt_ + \
            offset_ + \
            g_force_

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        # "save_path": "graphs_and_models/fastjet/land/___",
        # "offline_graph_path": "offline_graphs/fastjet/land/___/0_100e_4950p.graph"
    }
}

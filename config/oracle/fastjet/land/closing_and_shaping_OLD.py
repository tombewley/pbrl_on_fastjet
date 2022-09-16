from torch import abs, clamp, pi
from torch.linalg import norm
from rlutils.rewards.interfaces import OracleInterface
from ....features.fastjet import *
from ....features.fastjet import _vec_to_target, _target_right, _cosim # Private

def oracle(s, a, ns):
    s, a, ns = preprocessor(s, a, ns) # NOTE: required to normalise attitude vectors

    abs_lr_offset_ = abs(lr_offset(s, a, ns, None))

    is_closing_xz = (delta_dist_xz(s, a, ns, {"dist_xz": dist_xz(s, a, ns, None)}) < 0.).float()
    # NOTE: This just replicates delta_alt_error
    is_closing_y =  (abs(ns[...,1] - ns[...,20]) - abs(s[...,1] - s[...,20]) < 0.).float()
    on_line =       (abs_lr_offset_ < 10.).float()
    pitch_error_ =  clamp(0.5 - abs(pitch(s, a, ns, None) - target_pitch(s, a, ns, None)), 0.)
    abs_roll_ =     clamp(pi/2 - abs(roll(s, a, ns, None)), 0.)
    hdg_error_ =    clamp(0.5 - hdg_error(s, a, ns, None), 0.)
    alt_ =          clamp(20. - abs(alt(s, a, ns, None)), 0.)
    offset_ =       clamp(60. - abs_lr_offset_, 0.)
    g_force_ =      clamp(4. - g_force(s, a, ns, None), 0.)

    return 1.    * is_closing_xz + \
           2.    * is_closing_y + \
           1.    * on_line + \
           0.5   * pitch_error_ + \
           1.    * abs_roll_ + \
           1.    * hdg_error_ + \
           0.05  * alt_ + \
           0.05  * offset_ + \
           0.1   * g_force_

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface,
            "oracle": oracle
        },
        "save_path": "graphs_and_models/fastjet/land/closing_and_shaping",
        "offline_graph_path": "offline_graphs/fastjet/land/closing_and_shaping/0_200e_1000p_OLD.graph"
    }
}

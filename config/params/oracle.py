from rlutils.observers.pbrl.interfaces import OracleInterface
from ..oracles import *

P = {
    "pbrl": {
        "interface": {
            "class": OracleInterface, 
            "oracle": {
                "target_pose_tree":      target_pose_tree,
                "target_pose_linear":    target_pose_linear,
                "dist_only":             dist_only,
                "dist_closing_uperr":    dist_closing_uperr,
                "dist_closing_uperr_v2": dist_closing_uperr_v2,
                "g_force":               g_force,
                "pitch":                 pitch,
            },
        }
    }
}
from rlutils.observers.pbrl.interfaces import OracleInterface
from ..oracles import *

P = {
    "pbrl": {
        "interface": {
            "kind": OracleInterface, 
            "oracle": [
                target_pose_tree,        # 0
                target_pose_linear,      # 1
                negative_dist_to_target, # 2
                dist_closing_uperr,      # 3
                dist_closing_uperr_v2    # 4
            ]
        },
    }
}
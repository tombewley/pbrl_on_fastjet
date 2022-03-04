from rlutils.observers.pbrl.interfaces import OracleInterface
from ..oracles import *

P = {
    "pbrl": {
        "interface": {
            "kind": OracleInterface, 
            "oracle": target_pose_tree
        },
        "reward_source": "oracle"
    }
}
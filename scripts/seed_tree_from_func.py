from rlutils.observers.pbrl.models import RewardTree
from hyperrectangles.rules import rules


model = RewardTree({
    "featuriser": {
        "feature_names": ["x_pos", "y_pos"]
    }
})

def func():
    if y_pos < 1:
        if x_pos < 1: 
            return
        else:
            return
    else:
        return

model.tree = model.make_tree(func)

import numpy as np
model.tree.space.data = np.array([
    [0,0,0,0],
    [0,0,0,1],
    [0,0,0,2]
])
model.tree.populate()
model.tree = model.tree

print(rules(model.tree, pred_dims="reward", dims_as_indices=False))
print(model.r)
print(model.var)
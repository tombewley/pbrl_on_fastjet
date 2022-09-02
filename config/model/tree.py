from rlutils.rewards.models import RewardTree

P = {
    "pbrl": {
        "reward_source": "model",
        "model": {
            "class":              RewardTree,
            "num_from_queue":     float("inf"),
            "min_samples_leaf":   1,
            "m_max":              100,
            "preference_eqn":     "bradley-terry",
            "store_all_qual":     False,
            "split_dim_entropy":  0., 
            "nodewise_partition": False,

            #                       0      1      2      3      4      5      6      7      8      9      10      11     12
            "split_by_preference": [False, False, False, False, False, False, False, True,  True,  True,  True,  True,   True   ],
            "loss_func":           ["bce", "bce", "bce", "0-1", "0-1", "0-1", "0-1", "bce", "0-1", "0-1", "0-1", "0-1",  "0-1"  ],
            "alpha":               [0.01,  0.001, 0.,    0.01,  0.001, 0.,    0.,    0.,    0.,    0.,    0.,    0.,     0.,    ],
            "prune_ratio":         [None,  None,  None,  None,  None,  None,  0.5,   None,  None,  0.5,   0.5,   0.5,    0.5    ],
            "trees_per_update":    [1,     1,     1,     1,     1,     1,     5,     1,     1,     1,     5,     1,      5      ],
            "forest_size":         [1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     1,     10,     10     ],
            "sort_forest_by":      ["age", "age", "age", "age", "age", "age", "age", "age", "age", "age", "age", "loss", "loss" ]
        }
    }
}

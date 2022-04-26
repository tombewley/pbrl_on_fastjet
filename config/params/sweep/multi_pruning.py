P = {
    "pbrl": {
        "model": {
            "loss_func":              ["bce", "bce", "bce", "bce", "0-1", "bce"],
            "trees_per_update":       [1,     1,     1,     1,     1,     5    ],
            "prune_ratio":            [None,  0.5,   0.5,   0.9,   0.5,   0.5  ],
            "post_populate_with_all": [False, False, True,  False, False, False],
            "alpha":                  [0.001, 0,     0,     0,     0,     0    ]
        }
    }
}


P = {
    "deployment": {
        "agent": "pets"
    },
    "agent": {
        "pretrained_model": True,
        "num_random_steps": 0,
        "model_freq": 0,
        "rollout": {"horizon_params": ("constant", 10)},
        "gamma": 1,
        #                    0      1       2       3
        "cem_iterations":   [50,    10,     50,     10,     ],
        "cem_particles":    [60,    20,     60,     20,     ],
        "cem_elites":       [15,    5,      15,     5,      ],
        "cem_warm_start":   [False, False,  True,   True,   ],
        "cem_alpha":        [0.5,   0.5,    0.5,    0.5,    ],
        "cem_temperature":  [0.,    0.,     0.5,    0.5,    ]
    }
}

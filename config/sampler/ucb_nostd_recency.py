raise Exception("IS UNIFORM BETTER?")
P = {
    "pbrl": {
        "sampler": {
            "weight": "ucb", 
            "num_std": 0,
            "recency_constraint": True,
            "probabilistic": True
        }
    }
}

P = {
    "pbrl": {
        "sampler": {
            "weight": "entropy",
            "preference_eqn": "thurstone", # NOTE: Cannot use thurstone with B-T net
            "recency_constraint": True,
            "probabilistic": True
        }
    }
}

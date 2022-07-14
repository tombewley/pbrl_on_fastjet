from torch import device, load
from torch.cuda import is_available

P = {
    "deployment": {
        "agent": "pets",
    },
    "agent": {
        "pretrained_model": load(
            "pretrained_dynamics/follow_v1.dynamics"
            # "pretrained_dynamics/target_hard_v1.dynamics"
            # "pretrained_dynamics/chase_v1.dynamics"
            , map_location=device("cuda" if is_available() else "cpu")),

        # "replay_capacity": 5e5,
        "num_random_steps": 0,
        # "batch_ratio": 1,

        "model_freq": 0,
        # "batch_size": 256,
        # "lr_model": 1e-3,

        "cem_iterations": 50,
        "cem_particles": 60,
        "cem_elites": 15,
        "alpha": 0.5,
        "gamma": 1,#0.99,

        "rollout": {
            "horizon_params": ("constant", 10)
        }
    }
}

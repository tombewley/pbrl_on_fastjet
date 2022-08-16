from torch.linalg import norm
from fastjet.tasks.target_hard import CONFIG

SKIP_FRAMES = 25

def termination(_, __, next_states):
    return norm(next_states[...,0:3] - next_states[...,19:22], axis=-1) < CONFIG["radius"]

P = {
    "deployment": {
        "env": "FastJet-v0",
        "task": "target_hard",
        "episode_time_limit": 750 / SKIP_FRAMES,
        "skip_frames": SKIP_FRAMES
    },
    "agent": {
        "termination": termination
    }
}

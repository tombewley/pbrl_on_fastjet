SKIP_FRAMES = 25

def termination(_, __, next_states):
    return next_states[...,1] <= 0.5

P = {
    "deployment": {
        "env": "FastJet-v0",
        "task": "land",
        "episode_time_limit": 600 / SKIP_FRAMES,
        "skip_frames": SKIP_FRAMES,
        "camera_angle": "outside_parallel_skew"
    },
    "agent": {
        "termination": termination
    },
    "pbrl": {
        "model": {
            "negative_rewards": False
        }
    }
}

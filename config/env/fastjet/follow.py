SKIP_FRAMES = 25

P = {
    "deployment": {
        "env": "FastJet-v0",
        "task": "follow",
        "episode_time_limit": 500 / SKIP_FRAMES,
        "skip_frames": SKIP_FRAMES,
        "camera_angle": "outside_parallel_skew"
    },
    "pbrl": {
        "model": {
            "negative_rewards": False
        }
    }
}
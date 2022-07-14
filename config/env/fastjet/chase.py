SKIP_FRAMES = 25

P = {
    "deployment": {
        "env": "FastJet-v0",
        "task": "chase",
        "project_name": "fastjet-chase",
        "episode_time_limit": 500 / SKIP_FRAMES,
        "skip_frames": SKIP_FRAMES
    },
    "pbrl": {
        "save_path": "graphs_and_models/fastjet/chase"
    }
}
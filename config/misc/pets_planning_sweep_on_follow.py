SKIP_FRAMES = 25

P = {
    "deployment": {
        "env": "FastJet-v0",
        "task": "follow",
        "num_episodes": 100,
        "episode_time_limit": 500 / SKIP_FRAMES,
        "skip_frames": SKIP_FRAMES,
        "render_freq": 0,
        "wandb_monitor": True,
    }
}

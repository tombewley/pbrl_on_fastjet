import wandb
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

raise Exception("Use rlutils.experiments.wandb")

PROJECT = "fastjet-follow"
SAVE_DIR = "results/pets_vs_sac"

FILTER_TAG = None # "test.method"
FILTER_KEY = "name"
FILTER_VALUES = [
    "stilted-fog-57",
    "dulcet-planet-58",
    "dark-lake-59",
    "ancient-serenity-60",
    "brisk-dream-61",
    "kind-deluge-62",
    "summer-darkness-63",
    "dark-flower-64",
    "floral-jazz-65",
    "prime-dream-66",
    "denim-haze-24"
]
METRIC_KEYS = [
    # "reward_sum_oracle",
    # "reward_sum_model",
    # "num_leaves",
    # "preference_loss",
    "_runtime"
]

EP_MAX = None # 20000

api = wandb.Api()
key_split = FILTER_KEY.split(".")
os.makedirs(SAVE_DIR, exist_ok=True)

data = {value: {"run_names": [], "metrics": []} for value in FILTER_VALUES}    
for k, run in enumerate(tqdm(api.runs(f"tombewley/{PROJECT}"))):
    if FILTER_TAG is None or FILTER_TAG in run.tags:
        value = run.config; value.update({"name": run.name})
        for key in key_split: 
            try: value = value[key]
            except: value = None; break
        if value in FILTER_VALUES:
            print(run.name, value)
            data[value]["run_names"].append(run.name)
            data[value]["metrics"].append([])
            # run.scan_history() prevents downsampling to 500 rows as in run.history()
            for i, row in enumerate(tqdm(run.scan_history(), total=EP_MAX, leave=False)):
                data[value]["metrics"][-1].append([row[m] if m in row else np.nan for m in METRIC_KEYS])
                if EP_MAX is not None and (i+1) == EP_MAX: break
            if EP_MAX is not None and (i+1) < EP_MAX: 
                data[value]["metrics"][-1] += [[np.nan for _ in METRIC_KEYS]] * (EP_MAX - (i+1))

for value in FILTER_VALUES:
    for m, data_m in zip(METRIC_KEYS, np.transpose(np.array(data[value]["metrics"]))):
        df = pd.DataFrame(data_m, columns=data[value]["run_names"])
        df.to_csv(f"{SAVE_DIR}/{FILTER_TAG+'_' if FILTER_TAG is not None else ''}{FILTER_KEY}={value}_{m}.csv", index=False)

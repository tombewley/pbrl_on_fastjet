from torch import device, load
from torch.cuda import is_available

P = {
    "deployment": {
        "agent": "pets",
    },
    "agent": {
        # "net_model": [(None, 128), "R", (128, 128), "R", (128, None)],
        "net_model": [(None, 200), "R", (200, 200), "R", (200, 200), "R", (200, 200), "R", (200, None)],
        "input_normaliser": "box_bounds",
        "probabilistic": False,
        "ensemble_size": 5,
        
        "batch_size": 256,
        "lr_model": 1e-3,

        "model_freq": 0, # NOTE: Collecting all data then updating at the end

        "reward": None,

        # NOTE: Other parameters are unused in random mode so leave as defaults
    }
}

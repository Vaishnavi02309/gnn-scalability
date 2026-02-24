"""Purpose: make results reproducible. Without it, each run uses different splits, 
different sampled nodes, different weights → results become noisy and not comparable."""


import os
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Best-effort determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

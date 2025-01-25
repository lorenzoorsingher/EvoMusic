from diffusers.utils.testing_utils import enable_full_determinism
from EvoMusic.application import EvoMusic

import torch
import numpy as np
import random

import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    enable_full_determinism()

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.seed_all()
        
    app = EvoMusic("config.yaml")
    
    app.generation_loop(0)
    
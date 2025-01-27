from diffusers.utils.testing_utils import enable_full_determinism
from EvoMusic.application import EvoMusic

import torch
import numpy as np
import random
import argparse
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    enable_full_determinism()
    # create argparser
    parser = argparse.ArgumentParser(description="Evolve your own personalized music")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    config_path = parser.parse_args().config_path
    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.seed_all()

    app = EvoMusic(config_path)

    app.generation_loop(0)

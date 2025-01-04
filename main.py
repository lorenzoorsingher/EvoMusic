import torch

from usrapprox.usrapprox.models.aligner_v2 import AlignerWrapper
from usrapprox.usrapprox.models.usr_emb import UsrEmb
from usrapprox.usrapprox.utils.utils import Categories
from usrembeds.models.model import AlignerV2

# from usrapprox.models.probabilistic import calculate_logits, probabilistic_model_torch
# from usrapprox.utils.utils import Categories

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

if __name__ == "__main__":
    # alignerv2 = AlignerWrapper()
    user_embedder = UsrEmb()

    exit(0)
    

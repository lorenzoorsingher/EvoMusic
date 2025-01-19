
import torch
from tqdm import tqdm
from usrapprox.usrapprox.models.aligner_v2 import AlignerV2Wrapper
from usrapprox.usrapprox.models.usr_emb import UsrEmb


def train_one_epoch_synthetic(model: UsrEmb, synthetic_users: AlignerV2Wrapper, dataloader : torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, log:bool = True, log_frequency: int = 100, synthethic_user_id: int = 0):
    model.train()

    losses = []
    for i, batch in enumerate(tqdm(dataloader)):
        
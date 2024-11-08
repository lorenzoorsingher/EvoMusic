import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset, ContrDataset
from models.model import Aligner
from utils import get_args, load_model
import torch.nn as nn

from torch import optim

from tqdm import tqdm

from dotenv import load_dotenv
import os
import wandb
import datetime

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


if __name__ == "__main__":

    args = get_args()

    LOAD = "usrembeds/checkpoints/run_20241107_201542_best.pt"

    model_state, config, opt_state = load_model(LOAD)
    experiments = [config]

    BATCH_SIZE = config["batch_size"]
    EMB_SIZE = config["emb_size"]
    MUSIC_EMB_SIZE = config["prj_size"]
    NEG = config["neg_samples"]
    TEMP = config["temp"]
    LT = config["learnable_temp"]
    MUL = config["multiplier"]
    WEIGHT = config["weight"]
    PRJ = config["prj"]

    membs_path = "usrembeds/data/embeddings/batched"
    stats_path = "clean_stats.csv"
    save_path = "usrembeds/checkpoints"

    dataset = ContrDataset(
        membs_path,
        stats_path,
        nneg=NEG,
        multiplier=MUL,
        transform=None,
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    NUSERS = dataset.nusers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=True
    )

    model = Aligner(
        n_users=NUSERS,
        emb_size=EMB_SIZE,
        prj_size=MUSIC_EMB_SIZE,
        prj_type=PRJ,
        lt=LT,
        temp=TEMP,
    ).to(DEVICE)

    model.load_state_dict(model_state)
    # opt = optim.AdamW(model.parameters(), lr=0.001)
    # opt.load_state_dict(opt_state)

    config = {
        "emb_size": EMB_SIZE,
        "batch_size": BATCH_SIZE,
        "neg_samples": NEG,
        "temp": TEMP,
        "learnable_temp": LT,
        "multiplier": MUL,
        "weight": WEIGHT,
        "prj": PRJ,
        "nusers": NUSERS,
        "prj_size": MUSIC_EMB_SIZE,
    }

    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #                     optimizer, "min", factor=0.2, patience=PAT // 2
    #                 )

    model.eval()

    positives = torch.empty(0).to(DEVICE)
    negatives = torch.empty(0).to(DEVICE)

    trues = 0
    numel = 0
    for tracks in tqdm(train_dataloader):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        allemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, _ = model(idx, allemb)

        # breakpoint()
        posemb_out = embs[:, 0, :].unsqueeze(dim=1)
        negemb_out = embs[:, 1:, :]

        # breakpoint()
        out = urs_x.unsqueeze(1)
        # breakpoint()

        # breakpoint()
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        possim = cos(out, posemb_out).squeeze(1)

        out = out.repeat(1, negemb_out.shape[1], 1)
        negsim = cos(out, negemb_out)

        negsim = negsim.view(-1, negemb_out.shape[1])
        negflat = negsim.flatten()
        # breakpont()
        positives = torch.cat((positives, possim))
        negatives = torch.cat((negatives, negflat))

        comp = possim.unsqueeze(1).repeat(1, 20) > negsim

        trues += comp.sum().item()
        numel += comp.numel()

        print(f"Accuracy: {trues / numel}")

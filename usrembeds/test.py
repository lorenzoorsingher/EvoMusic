import torch
import torchopenl3
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from tqdm import tqdm

from datautils.dataset import ContrDatasetMERT
from models.model import Aligner


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


if __name__ == "__main__":

    # load model and config from checkpoint
    LOAD = "usrembeds/checkpoints/MERT_full_gating_linear_checkpoint.pt"
    model_state, setup, _ = Aligner.load_model(LOAD)

    default = {
        "emb_size": 256,
        "batch_size": 64,
        "neg_samples": 20,
        "temp": 0.5,
        "learnable_temp": False,
        "multiplier": 10,
        "weight": 0,
        "prj": "linear",
        "aggr": "gating",
        "nusers": 1000,
        "prj_size": 768,
        "drop": 0.25,
        "lr": 0.001,
        "encoder": "MERT",
    }

    config = default.copy()
    config = {**config, **setup}

    BATCH_SIZE = config["batch_size"]
    EMB_SIZE = config["emb_size"]
    NEG = config["neg_samples"]
    TEMP = config["temp"]
    LT = config["learnable_temp"]
    MUL = config["multiplier"]
    WEIGHT = config["weight"]
    PRJ = config["prj"]
    AGGR = config["aggr"]
    DROP = config["drop"]
    LR = config["lr"]
    ENCODER = config["encoder"]
    MUSIC_EMB_SIZE = config["prj_size"]

    membs_path = "usrembeds/data/embeddings/embeddings_full_split"
    stats_path = "clean_stats.csv"
    save_path = "usrembeds/checkpoints"

    dataset = ContrDatasetMERT(
        membs_path,
        stats_path,
        nneg=NEG,
        multiplier=MUL,
        transform=None,
    )

    _, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    NUSERS = dataset.nusers

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=8,
    )

    model = Aligner(
        n_users=NUSERS,
        emb_size=EMB_SIZE,
        prj_size=MUSIC_EMB_SIZE,
        prj_type=PRJ,
        aggragation=AGGR,
        lt=LT,
        temp=TEMP,
        drop=DROP,
    ).to(DEVICE)

    model.load_state_dict(model_state)

    model.eval()

    positives = torch.empty(0)
    negatives = torch.empty(0)

    trues = 0
    numel = 0
    for tracks in tqdm(val_dataloader):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        allemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, temp = model(idx, allemb)

        # breakpoint()
        posemb_out = embs[:, 0, :].unsqueeze(dim=1)
        negemb_out = embs[:, 1:, :]

        # breakpoint()
        out = urs_x.unsqueeze(1)

        # breakpoint()
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        possim = cos(out, posemb_out).squeeze(1).cpu().detach()

        out = out.repeat(1, negemb_out.shape[1], 1)
        negsim = cos(out, negemb_out).cpu().detach()

        negsim = negsim.view(-1, negemb_out.shape[1])
        negflat = negsim.flatten()

        positives = torch.cat((positives, possim))
        negatives = torch.cat((negatives, negflat))

        # accumulate number of correct predictions
        comp = possim.unsqueeze(1).repeat(1, 20) > negsim

        trues += comp.sum().item()
        numel += comp.numel()

    # compute ROC AUC, PR AUC, and accuracy
    np_pos = positives.cpu().detach().numpy()
    np_neg = negatives.cpu().detach().numpy()
    scores = np.concatenate((np_pos, np_neg))
    labels = [1] * len(np_pos) + [0] * len(np_neg)

    fpr, tpr, thresholds = roc_curve(labels, scores)

    roc_auc = roc_auc_score(labels, scores)

    pr_auc = average_precision_score(labels, scores)

    print(f"ROC AUC: {round(roc_auc,2)} PR AUC: {round(pr_auc,2)}")
    print(f"Accuracy: {round(trues / numel,2)}")

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Embedding Similarity")
    plt.legend(loc="best")
    plt.show()

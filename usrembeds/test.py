import torch
import torchopenl3
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from tqdm import tqdm

from datautils.dataset import ContrDataset
from models.model import Aligner


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


if __name__ == "__main__":

    # load model and config from checkpoint
    LOAD = "usrembeds/checkpoints/run_20241107_201542_best.pt"
    model_state, config, _ = Aligner.load_model(LOAD)

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

    # load dataset and dataloader
    dataset = ContrDataset(
        membs_path,
        stats_path,
        nneg=NEG,
        multiplier=MUL,
        transform=None,
    )

    _, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    NUSERS = dataset.nusers

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=True
    )

    # load model
    model = Aligner(
        n_users=NUSERS,
        emb_size=EMB_SIZE,
        prj_size=MUSIC_EMB_SIZE,
        prj_type=PRJ,
        lt=LT,
        temp=TEMP,
    ).to(DEVICE)

    model.load_state_dict(model_state)

    model.eval()

    positives = torch.empty(0).to(DEVICE)
    negatives = torch.empty(0).to(DEVICE)

    trues = 0
    numel = 0
    for tracks in tqdm(val_dataloader):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb, weights = tracks
        breakpoint()
        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        allemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, _ = model(idx, allemb)

        # separate positive and negative embeddings after projection
        posemb_out = embs[:, 0, :].unsqueeze(dim=1)
        negemb_out = embs[:, 1:, :]

        # compute cosine similarity between user embeddings and positive/negative embeddings
        out = urs_x.unsqueeze(1)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        possim = cos(out, posemb_out).squeeze(1)

        out = out.repeat(1, negemb_out.shape[1], 1)
        negsim = cos(out, negemb_out)

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

    print(f"ROC AUC: {roc_auc}")
    print(f"PR AUC: {pr_auc}")
    print(f"Accuracy: {trues / numel}")

    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Embedding Similarity")
    plt.legend(loc="best")
    plt.show()

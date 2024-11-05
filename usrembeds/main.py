import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset, ContrDataset
from models.model import Aligner
from utils import get_args
import torch.nn as nn

from torch import optim

from tqdm import tqdm

from dotenv import load_dotenv
import os
import wandb
import datetime

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


def weighted_contrastive_loss(out, possim, negsim, weights, loss_weight, temp=0.07):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    possim = cos(out, posemb)

    out = out.repeat(1, negemb.shape[1], 1)
    negsim = cos(out, negemb)

    # breakpoint()
    logits = torch.cat((possim, negsim), dim=1) / temp
    exp = torch.exp(logits)
    loss = -torch.log(exp[:, 0] / torch.sum(exp, dim=1))

    loss = loss * ((weights * loss_weight) + 1)

    loss = torch.mean(loss)
    return loss


def eval_loop(model, val_loader):

    model.eval()

    losses = []

    correct = 0
    total = 0
    for tracks in tqdm(val_loader):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)

        ellemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs = model(idx, ellemb)

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

        mean_negsim = torch.mean(negsim, dim=1)

        correct += (possim > mean_negsim).sum().item()
        total += possim.shape[0]
        # breakpoint()
        # logits = torch.cat((possim, negsim), dim=1) / 0.07
        # exp = torch.exp(logits)
        # loss = -torch.log(exp[:, 0] / torch.sum(exp, dim=1))
        # loss = torch.mean(loss)

        # losses.append(loss.item())

        # print(np.mean(losses))
    model.train()
    return correct / total


def eval_auc_loop(model, val_loader):

    model.eval()

    positives = torch.empty(0).to(DEVICE)
    negatives = torch.empty(0).to(DEVICE)

    for tracks in tqdm(val_loader):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        ellemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, _ = model(idx, ellemb)

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

        positives = torch.cat((positives, possim))
        negatives = torch.cat((negatives, negflat))
    np_pos = positives.cpu().detach().numpy()
    np_neg = negatives.cpu().detach().numpy()
    scores = np.concatenate((np_pos, np_neg))
    labels = [1] * len(np_pos) + [0] * len(np_neg)
    # fpr, tpr, thresholds = roc_curve(labels, scores)

    roc_auc = roc_auc_score(labels, scores)
    # print(f"ROC AUC: {roc_auc}")

    # Calculate PR AUC
    pr_auc = average_precision_score(labels, scores)
    # print(f"Precision-Recall AUC: {pr_auc}")

    # import matplotlib.pyplot as plt

    # plt.plot(fpr, tpr, label="ROC Curve")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC Curve for Embedding Similarity")
    # plt.legend(loc="best")
    # plt.show()
    # breakpoint()
    # # scores = [possim] + negsim
    # # labels = [1] + [0] * len(negsim)

    model.train()
    return roc_auc, pr_auc


if __name__ == "__main__":

    args = get_args()

    BATCH_SIZE = args["batch"]
    EMB_SIZE = args["embeds"]
    NEG = args["neg"]
    SUBSET = args["subset"]
    TEMP = args["temp"]
    LT = args["learnable_temp"]
    MUL = args["multiplier"]
    WEIGHT = args["weight"]

    LOG = not args["no_log"]
    LOG_EVERY = 100

    HOP_SIZE = 0.2
    AUDIO_LEN = 5
    EPOCHS = 1000

    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    if LOG:
        wandb.init(
            # set the wandb project where this run will be logged
            project="BIO",
            name=run_name,
            config={
                "emb_size": EMB_SIZE,
                "batch_size": BATCH_SIZE,
                "neg_samples": NEG,
                "temp": TEMP,
                "learnable_temp": LT,
                "multiplier": MUL,
                "loss weight": WEIGHT,
            },
        )

    membs_path = "usrembeds/data/embeddings/batched"
    stats_path = "clean_stats.csv"
    save_path = "usrembeds/checkpoints"

    dataset = ContrDataset(
        membs_path,
        stats_path,
        nneg=NEG,
        multiplier=MUL,
        subset=SUBSET,
        transform=None,
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    NUSERS = dataset.nusers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    model = Aligner(
        n_users=NUSERS,
        emb_size=EMB_SIZE,
        prj_size=512,
        prj_type="bn",
        lt=LT,
    ).to(DEVICE)

    opt = optim.AdamW(model.parameters(), lr=0.001)

    best_auc = 0

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}")
        losses = []

        for itr, tracks in tqdm(enumerate(train_dataloader)):

            # [B]
            # [B, 1, EMB]
            # [B, NNEG, EMB]
            idx, posemb, negemb, weights = tracks

            idx = idx.to(DEVICE)
            posemb = posemb.to(DEVICE)
            negemb = negemb.to(DEVICE)
            weights = weights.to(DEVICE)

            opt.zero_grad()

            ellemb = torch.cat((posemb, negemb), dim=1)

            urs_x, embs, temp = model(idx, ellemb)

            # breakpoint()
            posemb_out = embs[:, 0, :].unsqueeze(dim=1)
            negemb_out = embs[:, 1:, :]

            # breakpoint()
            out = urs_x.unsqueeze(1)

            loss = weighted_contrastive_loss(
                out, posemb, negemb, weights, WEIGHT, temp=temp
            )
            # breakpoint()
            if itr % LOG_EVERY == 0 and LOG:
                if LT:
                    wandb.log({"loss": loss.item(), "temp": temp.item()})
                else:
                    wandb.log({"loss": loss.item()})

            losses.append(loss.item())

            loss.backward()
            opt.step()

        roc_auc, pr_auc = eval_auc_loop(model, val_dataloader)

        if roc_auc > best_auc:
            best_auc = roc_auc
            torch.save(model.state_dict(), f"{save_path}/{run_name}_best.pt")

        if LOG:
            wandb.log(
                {
                    "mean_loss": np.mean(losses),
                    "roc_auc": roc_auc,
                    "pr_auc": pr_auc,
                }
            )

        print(
            f"loss {round(np.mean(losses),3)} roc_auc {round(roc_auc,3)} pr_auc {round(pr_auc,3)}"
        )
        # print(loss)

if LOG:
    wandb.finish()

import torch
import os
import wandb
import datetime
import torchopenl3
import numpy as np

import torch.nn as nn
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from torch import optim
from tqdm import tqdm
from dotenv import load_dotenv

from datautils.dataset import MusicDataset, ContrDataset
from models.model import Aligner
from utils import get_args


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


def weighted_contrastive_loss(out, posemb, negemb, weights, loss_weight, temp=0.07):
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


def eval_auc_loop(model, val_loader, weight=0):

    model.eval()

    positives = torch.empty(0).to(DEVICE)
    negatives = torch.empty(0).to(DEVICE)
    val_losses = []
    for tracks in tqdm(val_loader):

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

        val_loss = weighted_contrastive_loss(
            out, posemb_out, negemb_out, weights, weight, temp=temp
        )

        val_losses.append(val_loss.item())

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

    return roc_auc, pr_auc, val_losses


def train_loop(model, train_loader, opt, weight, log=False, log_every=100):

    model.train()

    losses = []
    for itr, tracks in enumerate(tqdm(train_loader)):

        # [B]
        # [B, 1, EMB]
        # [B, NNEG, EMB]
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        opt.zero_grad()

        allemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, temp = model(idx, allemb)

        posemb_out = embs[:, 0, :].unsqueeze(dim=1)
        negemb_out = embs[:, 1:, :]

        out = urs_x.unsqueeze(1)

        loss = weighted_contrastive_loss(
            out, posemb_out, negemb_out, weights, weight, temp=temp
        )

        if itr % log_every == 0 and log:
            if LT:
                wandb.log({"loss": loss.item(), "temp": temp.item()})
            else:
                wandb.log({"loss": loss.item()})

        losses.append(loss.item())

        loss.backward()
        opt.step()
        # print(temp.item())
    return losses


if __name__ == "__main__":

    args = get_args()

    LOAD = args["load"]

    default = {
        "emb_size": 200,
        "batch_size": 16,
        "neg_samples": 20,
        "temp": 0.07,
        "learnable_temp": False,
        "multiplier": 10,
        "weight": 0,
        "prj": "bn",
        "nusers": 1000,
        "prj_size": 512,
    }

    if LOAD == "":
        print("[LOADER] Loading parameters from command line")
        experiments = [args]

    elif LOAD == "exp":
        print("[LOADER] Loading parameters from experiments set")
        experiments = [
            {
                "temp": 0.15,
                "learnable_temp": False,
            },
            {
                "temp": 0.15,
                "learnable_temp": True,
                "weight": 0.5,
            },
            {
                "temp": 1,
                "learnable_temp": True,
            },
        ]
    else:
        print("[LOADER] Loading parameters from checkpoint")
        model_state, config, opt_state = Aligner.load_model(LOAD)
        experiments = [config]

    LOG = not args["no_log"]
    LOG_EVERY = 100

    EPOCHS = 1000
    PAT = 6

    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    ##############################################################################

    for exp_num, exp in enumerate(experiments):

        print(f"[MAIN] Running experiment {exp_num+1} or {len(experiments)}")

        config = default.copy()
        config = config | exp

        BATCH_SIZE = config["batch_size"]
        EMB_SIZE = config["emb_size"]
        MUSIC_EMB_SIZE = config["prj_size"]
        NEG = config["neg_samples"]
        TEMP = config["temp"]
        LT = config["learnable_temp"]
        MUL = config["multiplier"]
        WEIGHT = config["weight"]
        PRJ = config["prj"]

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
        if LOG:
            wandb.init(
                project="BIO",
                name=run_name,
                config={
                    "emb_size": EMB_SIZE,
                    "batch_size": BATCH_SIZE,
                    "neg_samples": NEG,
                    "temp": TEMP,
                    "learnable_temp": LT,
                    "multiplier": MUL,
                    "weight": WEIGHT,
                    "prj": PRJ,
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
            transform=None,
        )

        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

        NUSERS = dataset.nusers
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            # pin_memory=True,
            num_workers=8,
        )
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
            lt=LT,
            temp=TEMP,
        ).to(DEVICE)

        if LOAD != "" and LOAD != "exp":
            model.load_state_dict(model_state)
            opt = optim.AdamW(model.parameters(), lr=0.001)
            opt.load_state_dict(opt_state)
        else:
            opt = optim.AdamW(model.parameters(), lr=0.001)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, "min", factor=0.2, patience=PAT // 2
        )

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

        print(f"[MAIN] Starting run {run_name}")
        print(f"[MAIN] With parameters:")
        for k, v in config.items():
            print(f"[MAIN] {k}: {v}")

        best_auc = 0
        pat = PAT

        for epoch in range(EPOCHS):

            print(f"Epoch {epoch}")

            losses = train_loop(model, train_dataloader, opt, WEIGHT, LOG, LOG_EVERY)
            roc_auc, pr_auc, val_losses = eval_auc_loop(model, val_dataloader, WEIGHT)

            scheduler.step(np.mean(val_losses))

            if LOG:
                wandb.log(
                    {
                        "mean_loss": np.mean(losses),
                        "val_loss": np.mean(val_losses),
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                    }
                )

            if roc_auc > best_auc:
                best_auc = roc_auc
                model_savefile = {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "config": config,
                }
                torch.save(model_savefile, f"{save_path}/{run_name}_best.pt")
                pat = PAT
            else:
                pat -= 1
                if pat == 0:
                    break
            print(
                f"loss {round(np.mean(losses),3)} roc_auc {round(roc_auc,3)} pr_auc {round(pr_auc,3)}"
            )
            # print(loss)

        if LOG:
            wandb.log(
                {
                    "best_auc": best_auc,
                }
            )
            wandb.finish()

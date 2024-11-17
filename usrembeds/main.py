import json
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

from datautils.dataset import ContrDatasetMERT, get_dataloaders
from models.model import Aligner, AlignerV2
from utils import get_args, gen_run_name


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!
print(f"Using {DEVICE}")


def weighted_contrastive_loss(out, posemb, negemb, weights, loss_weight, temp=0.07):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    possim = cos(out, posemb)

    out = out.repeat(1, negemb.shape[1], 1)
    negsim = cos(out, negemb)

    # breakpoint()
    logits = torch.cat((possim, negsim), dim=1) / temp
    exp = torch.exp(logits)
    denom = torch.sum(exp, dim=1) + 1e-6
    loss = -torch.log(exp[:, 0] / denom)

    loss = loss * ((weights * loss_weight) + 1)
    loss = torch.mean(loss)
    return loss


def eval_auc_loop(model, val_loader, weight=0):

    model.eval()

    positives = torch.empty(0)
    negatives = torch.empty(0)
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

        possim = cos(out, posemb_out).squeeze(1).cpu().detach()

        out = out.repeat(1, negemb_out.shape[1], 1)
        negsim = cos(out, negemb_out)

        negsim = negsim.view(-1, negemb_out.shape[1])
        negflat = negsim.flatten().cpu().detach()

        positives = torch.cat((positives, possim))
        negatives = torch.cat((negatives, negflat))
    np_pos = positives.numpy()
    np_neg = negatives.numpy()
    scores = np.concatenate((np_pos, np_neg))
    labels = [1] * len(np_pos) + [0] * len(np_neg)
    # fpr, tpr, thresholds = roc_curve(labels, scores)

    roc_auc = roc_auc_score(labels, scores)
    # print(f"ROC AUC: {roc_auc}")

    # Calculate PR AUC
    pr_auc = average_precision_score(labels, scores)

    return roc_auc, pr_auc, val_losses


def train_loop(model, train_loader, opt, weight, lt=False, log=False, log_every=100):

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
            out,
            posemb_out,
            negemb_out,
            weights,
            weight,
            temp=temp,
        )

        if itr % log_every == 0 and log:
            if lt:
                wandb.log({"loss": loss.item(), "temp": temp.item()})
            else:
                wandb.log({"loss": loss.item()})

        losses.append(loss.item())

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        loss.backward()
        opt.step()
        # print(temp.item())
    return losses


if __name__ == "__main__":

    args, args_dict = get_args()

    LOAD = args.load

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
        "pat": 10,
        "model": "AlignerV2",
    }

    if LOAD == "":
        print("[LOADER] Loading parameters from command line")
        experiments = [args_dict]

    elif LOAD == "exp":
        print("[LOADER] Loading parameters from experiments set")
        experiments = [
            {
                "aggr": "gating",
                "dropout": 0.25,
                "encoder": "MERT",
                "learnable_temp": True,
                "multiplier": 10,
                "neg_samples": 20,
                "prj": "linear",
                "temp": 0.5,
            }
        ]
    else:

        if LOAD.split(".")[-1] == "json":
            print("[LOADER] Loading parameters from json file")
            experiments = json.load(open(LOAD, "r"))
        elif LOAD.split(".")[-1] == "pt":
            print("[LOADER] Loading parameters from checkpoint")
            model_state, config, opt_state = Aligner.load_model(LOAD)
            experiments = [config]
        else:
            print("[LOADER] Error loading parameters, unknown file type")
            exit()

    LOG = not args.no_log
    LOG_EVERY = 100

    EPOCHS = 1000

    if LOG:
        load_dotenv()
        WANDB_SECRET = os.getenv("WANDB_SECRET")
        wandb.login(key=WANDB_SECRET)

    ##############################################################################

    for exp_num, exp in enumerate(experiments):

        print(f"[MAIN] Running experiment {exp_num+1} or {len(experiments)}")

        config = default.copy()
        config = {**config, **exp}

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
        PAT = config["pat"]

        embs_path = "usrembeds/data/embeddings/embeddings_full_split"
        stats_path = "usrembeds/data/clean_stats.csv"
        splits_path = "usrembeds/data/splits.json"
        save_path = "usrembeds/checkpoints"

        train_dataloader, val_dataloader, NUSERS = get_dataloaders(
            embs_path,
            stats_path,
            splits_path,
            NEG,
            MUL,
            BATCH_SIZE,
        )
        config["nusers"] = NUSERS

        if config["model"] == "Aligner":
            model_class = Aligner
        elif config["model"] == "AlignerV2":
            model_class = AlignerV2

        model = model_class(
            n_users=NUSERS,
            emb_size=EMB_SIZE,
            prj_size=MUSIC_EMB_SIZE,
            prj_type=PRJ,
            aggragation=AGGR,
            lt=LT,
            temp=TEMP,
            drop=DROP,
        ).to(DEVICE)

        if LOAD.split(".")[-1] == "pt":
            model.load_state_dict(model_state)
            opt = optim.AdamW(model.parameters(), lr=LR)
            opt.load_state_dict(opt_state)
        else:
            opt = optim.AdamW(model.parameters(), lr=LR)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            opt, "max", factor=0.2, patience=3
        )

        run_name = gen_run_name()
        if LOG:
            wandb.init(
                project="BIO",
                name=run_name,
                config=config,
            )

        print(f"[MAIN] Starting run {run_name}")
        print(f"[MAIN] With parameters:")
        for k, v in config.items():
            print(f"[MAIN] {k}: {v}")

        best_auc = 0
        pat = PAT

        for epoch in range(EPOCHS):

            print(f"Epoch {epoch}")

            losses = train_loop(
                model, train_dataloader, opt, WEIGHT, LT, LOG, LOG_EVERY
            )
            roc_auc, pr_auc, val_losses = eval_auc_loop(model, val_dataloader, WEIGHT)

            scheduler.step(roc_auc)

            if LOG:
                wandb.log(
                    {
                        "mean_loss": np.mean(losses),
                        "val_loss": np.mean(val_losses),
                        "roc_auc": roc_auc,
                        "pr_auc": pr_auc,
                        "lr": opt.param_groups[0]["lr"],
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
                f"loss {round(np.mean(losses),3)} roc_auc {round(roc_auc,3)} pr_auc {round(pr_auc,3)} PAT {pat}"
            )
            # print(loss)

        if LOG:
            wandb.log(
                {
                    "best_auc": best_auc,
                }
            )
            wandb.finish()

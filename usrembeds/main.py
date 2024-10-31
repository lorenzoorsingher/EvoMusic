import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset, StatsDataset, ContrDataset
from models.model import Aligner
from utils import plot_music_batch
import torch.nn as nn

from torch import optim

from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


def contrastive_loss(out, possim, negsim, temp=0.07):
    # breakpoint()
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    possim = cos(out, posemb)

    out = out.repeat(1, negemb.shape[1], 1)
    negsim = cos(out, negemb)

    logits = torch.cat((possim, negsim), dim=1) / temp
    exp = torch.exp(logits)
    loss = -torch.log(exp[:, 0] / torch.sum(exp, dim=1))
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
        out = model(idx)

        out = out.unsqueeze(1)
        # breakpoint()

        # breakpoint()
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)

        possim = cos(out, posemb).squeeze(1)

        out = out.repeat(1, negemb.shape[1], 1)
        negsim = cos(out, negemb)

        negsim = negsim.view(-1, negemb.shape[1])

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


if __name__ == "__main__":

    BATCH_SIZE = 16
    EMB_SIZE = 512
    HOP_SIZE = 0.2
    AUDIO_LEN = 5

    music_path = "../scraper/music"
    membs_path = "usrembeds/data/embeddings/embeddings_1000.json"
    stats_path = "../scraper/data/clean_stats.csv"

    dataset = ContrDataset(
        membs_path,
        stats_path,
        nneg=32,
        multiplier=1,
        transform=None,
    )

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    NUSERS = dataset.nusers
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=16, shuffle=True
    )

    model = Aligner(
        n_users=NUSERS,
        emb_size=20,
        prj_size=EMB_SIZE,
    ).to(DEVICE)

    EPOCHS = 10

    opt = optim.AdamW(model.parameters(), lr=0.001)
    TEMP = 0.07

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}")
        losses = []
        for tracks in tqdm(train_dataloader):

            # [B]
            # [B, 1, EMB]
            # [B, NNEG, EMB]
            idx, posemb, negemb = tracks

            idx = idx.to(DEVICE)
            posemb = posemb.to(DEVICE)
            negemb = negemb.to(DEVICE)
            opt.zero_grad()
            out = model(idx)

            out = out.unsqueeze(1)

            loss = contrastive_loss(out, posemb, negemb)

            losses.append(loss.item())

            loss.backward()
            opt.step()
        val_acc = eval_loop(model, val_dataloader)

        print(f"loss {np.mean(losses)} val_acc {round(val_acc,3)}")
        # print(loss)

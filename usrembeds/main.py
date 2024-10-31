import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset, StatsDataset, ContrDataset
from models.model import Aligner
from utils import plot_music_batch
import torch.nn as nn

from torch import optim

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


def constrstive_loss(possim, negsim, temp=0.07):
    # breakpoint()

    possim = cos(out, posemb)

    out = out.repeat(1, negemb.shape[1], 1)
    negsim = cos(out, negemb)

    logits = torch.cat((possim, negsim), dim=1) / TEMP
    exp = torch.exp(logits)
    loss = -torch.log(exp[:, 0] / torch.sum(exp, dim=1))
    loss = torch.mean(loss)


if __name__ == "__main__":

    BATCH_SIZE = 16
    EMB_SIZE = 512
    HOP_SIZE = 0.2
    AUDIO_LEN = 5

    music_path = "../scraper/music"
    membs_path = "usrembeds/data/embeddings/embeddings_1000.json"
    stats_path = "../scraper/data/clean_stats.csv"

    dataset = ContrDataset(membs_path, stats_path, nneg=32, transform=None)
    NUSERS = dataset.nusers
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = Aligner(NUSERS, EMB_SIZE).to(DEVICE)

    EPOCHS = 10

    cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    opt = optim.Adam(model.parameters(), lr=0.001)
    TEMP = 0.07

    for epoch in range(EPOCHS):

        print(f"Epoch {epoch}")
        losses = []
        for tracks in dataloader:

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

            loss = constrstive_loss(out, posemb, negemb)

            losses.append(loss.item())

            loss.backward()
            opt.step()

        print(np.mean(losses))
        # print(loss)

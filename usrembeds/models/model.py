import torch

import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    def __init__(self, n_users, emb_size, prj_size, prj_type="linear"):

        super(Aligner, self).__init__()

        self.prj_type = prj_type

        self.users = nn.Embedding(n_users, emb_size)

        self.fc1 = nn.Linear(emb_size, 4096)
        self.fc2 = nn.Linear(4096, prj_size)

        self.ln = nn.LayerNorm(prj_size)
        self.bn = nn.BatchNorm1d(prj_size)
        self.fc3 = nn.Linear(prj_size, prj_size)

    def forward(self, idx, embs):

        # idx, embs = x

        user_embs = self.users(idx)

        urs_x = F.relu(self.fc1(user_embs))
        urs_x = self.fc2(urs_x)

        if self.prj_type == "linear":
            embs = self.fc3(embs)
        elif self.prj_type == "ln":
            embs = self.ln(embs)
        elif self.prj_type == "bn":
            # TODO: not sure this is the right way
            embs = embs.permute(0, 2, 1)
            embs = self.bn(embs)
            embs = embs.permute(0, 2, 1)

        return urs_x, embs


class TunedAligner(nn.Module):
    def __init__(self, n_users, emb_size, prj_size, prj_type="bn", temp=0.07):

        super(TunedAligner, self).__init__()

        self.prj_type = prj_type

        self.users = nn.Embedding(n_users, emb_size)

        self.fc1 = nn.Linear(emb_size, 4096)
        self.fc2 = nn.Linear(4096, prj_size)

        self.ln = nn.LayerNorm(prj_size)
        self.bn = nn.BatchNorm1d(prj_size)
        self.fc3 = nn.Linear(prj_size, prj_size)

        self.temp = nn.Parameter(torch.tensor(temp))

    def forward(self, idx, embs):

        # idx, embs = x

        user_embs = self.users(idx)

        urs_x = F.relu(self.fc1(user_embs))
        urs_x = self.fc2(urs_x)

        if self.prj_type == "linear":
            embs = self.fc3(embs)
        elif self.prj_type == "ln":
            embs = self.ln(embs)
        elif self.prj_type == "bn":
            # TODO: not sure this is the right way
            embs = embs.permute(0, 2, 1)
            embs = self.bn(embs)
            embs = embs.permute(0, 2, 1)

        return urs_x, embs, self.temp

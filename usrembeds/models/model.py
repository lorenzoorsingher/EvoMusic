import torch

import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    def __init__(self, n_users, emb_size, prj_size):

        super(Aligner, self).__init__()

        self.users = nn.Embedding(n_users, emb_size)

        self.fc1 = nn.Linear(emb_size, 4096)
        self.fc2 = nn.Linear(4096, prj_size)

    def forward(self, x):

        user_embs = self.users(x)

        x = F.relu(self.fc1(user_embs))
        x = self.fc2(x)

        return x

import torch

import torch.nn as nn
import torch.nn.functional as F


class Aligner(nn.Module):
    def __init__(
        self,
        n_users,
        emb_size,
        prj_size=512,
        prj_type="linear",
        lt=False,
        temp=0.07,
        drop=0.35,
    ):

        super(Aligner, self).__init__()

        self.prj_type = prj_type

        self.users = nn.Embedding(n_users, emb_size)

        self.fc1 = nn.Linear(emb_size, 2048)
        # self.fcmid = nn.Linear(4096, 4096)
        self.dropmid = nn.Dropout(drop)
        self.fc2 = nn.Linear(2048, prj_size)

        self.temp = temp
        if lt:
            print("[MODEL] Using learnable temperature")
            self.temp = nn.Parameter(torch.tensor(self.temp))

        if self.prj_type == "linear":
            self.ln2 = nn.LayerNorm(prj_size)
            self.fc3 = nn.Linear(prj_size, 4096)
            self.fc4 = nn.Linear(4096, prj_size)
        elif self.prj_type == "ln":
            self.ln = nn.LayerNorm(prj_size)
        elif self.prj_type == "bn":
            self.bn = nn.BatchNorm1d(prj_size)

        print(f"[MODEL] Using projection type: {prj_type}")

    def forward(self, idx, music_embs):

        user_embs = self.users(idx)

        usr_x = F.gelu(self.fc1(user_embs))
        usr_x = self.dropmid(usr_x)
        usr_x = self.fc2(usr_x)

        if self.prj_type == "linear":
            music_x = self.ln2(music_embs)
            music_x = F.gelu(self.fc3(music_x))
            music_x = self.fc4(music_x)
        elif self.prj_type == "ln":
            music_x = self.ln(music_embs)
        elif self.prj_type == "bn":
            music_x = music_embs.permute(0, 2, 1)
            music_x = self.bn(music_x)
            music_x = music_x.permute(0, 2, 1)

        return usr_x, music_x, self.temp

    def load_model(model_path, device="cuda"):
        model_savefile = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = model_savefile["model"]
        config = model_savefile["config"]
        opt_state = model_savefile["optimizer"]

        return state_dict, config, opt_state

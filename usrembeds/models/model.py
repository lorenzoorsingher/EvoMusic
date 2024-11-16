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
        aggragation="mean",
        lt=False,
        temp=0.07,
        drop=0.35,
    ):

        super(Aligner, self).__init__()

        self.prj_type = prj_type
        self.aggragation = aggragation

        if aggragation == "gating":
            # use attention to compute weights for each dimension of the music embeddings
            self.input_gate_ln = nn.LayerNorm(prj_size)
            self.gate = nn.Linear(prj_size*13, prj_size*13)
            self.gate_transform = nn.Linear(prj_size, prj_size)
            self.gate_norm = nn.LayerNorm(prj_size)
            self.drop_gate = nn.Dropout(drop)
        elif aggragation == "cross-attention":
            # use attention to compute weights for each dimension of the music embeddings
            self.input_gate_ln = nn.LayerNorm(prj_size)
            self.gate = nn.MultiheadAttention(prj_size, 6, dropout=drop, batch_first=True)
        elif aggragation == "weighted":
            self.weights = nn.Parameter(torch.ones(13))
        elif aggragation == "GRU":
            self.input_gate_ln = nn.LayerNorm(prj_size)
            self.gru = nn.GRU(prj_size, prj_size, 1, batch_first=True, dropout=drop)
        elif aggragation != "mean":
            raise ValueError(f"[MODEL] Invalid aggregation type: {aggragation}")

        self.users = nn.Embedding(n_users, emb_size)

        self.linear1 = nn.Linear(emb_size, 2048)
        self.fc1 = nn.Linear(emb_size, 2048)
        self.dropmid = nn.Dropout(drop)
        self.lnmid = nn.LayerNorm(2048)
        self.linear2 = nn.Linear(2048, prj_size)
        self.fc2 = nn.Linear(2048, prj_size)
        self.ln_usr = nn.LayerNorm(prj_size)

        self.temp = temp
        if lt:
            print("[MODEL] Using learnable temperature")
            self.temp = nn.Parameter(torch.tensor(self.temp))

        if self.prj_type == "linear":
            self.fc3 = nn.Linear(prj_size, prj_size)
            self.ln = nn.LayerNorm(prj_size)
        elif self.prj_type == "ln":
            self.ln = nn.LayerNorm(prj_size)
        elif self.prj_type == "bn":
            self.bn = nn.BatchNorm1d(prj_size)
        else:
            raise ValueError(f"[MODEL] Invalid projection type: {prj_type}")

        print(f"[MODEL] Using projection type: {prj_type}")

    def forward(self, idx, music_embs):

        user_embs = self.users(idx)

        usr_x = F.gelu(self.fc1(user_embs)) + self.linear1(user_embs)
        usr_x = self.lnmid(usr_x)
        usr_x = self.dropmid(usr_x)
        usr_x = F.gelu(self.fc2(usr_x)) + self.linear2(usr_x)
        usr_x = self.ln_usr(usr_x)

        batch_size = music_embs.size(0)
        N_EMB = music_embs.size(1)
    
        if self.aggragation == "mean":
            music_x = music_embs.mean(dim=2)
        elif self.aggragation == "weighted":
            music_x = (music_embs * self.weights).mean(dim=2)
        elif self.aggragation == "gating":
            music_embs = self.input_gate_ln(music_embs)
            # [batch, NPOS+NNEG, 13, emb_size] -> [batch, NPOS+NNEG, 13*emb_size]
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, -1)
            # compute weights for each dimension of the music embeddings to be normalized
            weights = self.gate(music_embs_reshaped).view(batch_size, N_EMB, 13, -1)
            # normalize over the 13 dimensions
            weights = F.softmax(weights, dim=2)
            # apply weighted sum to the music embeddings
            music_embs = self.gate_norm(music_embs + F.gelu(self.gate_transform(music_embs)))
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.drop_gate(music_x)
        elif self.aggragation == "cross-attention":
            music_embs = self.input_gate_ln(music_embs)
            # use user embeddings as query and music embeddings as key and value
            # urs [batch, prj_size] -> [batch * NPOS+NNEG, 1, prj_size]
            # music [batch, NPOS+NNEG, 13, emb_size] -> [batch * NPOS+NNEG, 13, emb_size]
            user_queries = usr_x.unsqueeze(1).repeat(1, N_EMB, 1).view(batch_size * N_EMB, 1, -1)
            music_vals = music_embs.view(batch_size * N_EMB, 13, -1)
            music_x, weights = self.gate(user_queries, music_vals, music_vals)
            music_x = music_x.view(batch_size, N_EMB, -1)
        elif self.aggragation == "GRU":
            music_embs = self.input_gate_ln(music_embs)
            music_gru_input = music_embs.view(batch_size * N_EMB, 13, -1)
            music_x, _ = self.gru(music_gru_input)
            music_x = music_x[:, -1, :].view(batch_size, N_EMB, -1)
        

        if self.prj_type == "linear":
            music_x = F.gelu(self.fc3(music_x)) + music_x
            music_x = self.ln(music_x)
        elif self.prj_type == "ln":
            music_x = self.ln(music_x)
        elif self.prj_type == "bn":
            music_x = music_x.permute(0, 2, 1)
            music_x = self.bn(music_x)
            music_x = music_x.permute(0, 2, 1)

        return usr_x, music_x, self.temp

    def load_model(model_path, device="cuda"):
        model_savefile = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = model_savefile["model"]
        config = model_savefile["config"]
        opt_state = model_savefile["optimizer"]

        return state_dict, config, opt_state

import torch

import torch.nn as nn
import torch.nn.functional as F


class AlignerV2(nn.Module):
    def __init__(
        self,
        n_users,
        emb_size,
        prj_size=512,
        hidden_size=2048,
        prj_type="linear",
        aggragation="mean",
        noise_level=0.0,
        lt=False,
        temp=0.07,
        drop=0.35,
    ):
        super(AlignerV2, self).__init__()

        self.cosine_similarity = nn.CosineSimilarity(dim=2, eps=1e-6)

        self.prj_type = prj_type
        self.prj_size = prj_size
        self.aggragation = aggragation
        self.noise_level = noise_level

        num_heads = 1

        if aggragation == "gating":
            # use attention to compute weights for each dimension of the music embeddings
            self.gate = nn.Linear(prj_size * 13, prj_size * 13)
            self.gate_transform = nn.Linear(prj_size, prj_size)
            self.gate_norm = nn.LayerNorm(prj_size)
            self.gate_out_norm = nn.LayerNorm(prj_size)
        elif aggragation == "gating-tanh":
            # use attention to compute weights for each dimension of the music embeddings
            self.gate = nn.Linear(prj_size * 13, prj_size * 13)
            self.gate_transform = nn.Linear(prj_size, prj_size)
            self.gate_norm = nn.LayerNorm(prj_size)
            self.gate_out_norm = nn.LayerNorm(prj_size)
        elif aggragation == "cross-attention":
            # use attention to compute weights for each dimension of the music embeddings
            self.gate = nn.MultiheadAttention(prj_size, num_heads, batch_first=True)
            self.gate_norm = nn.LayerNorm(prj_size)
        elif aggragation == "weighted":
            self.weights = nn.Parameter(torch.ones(13))
        elif aggragation == "GRU":
            self.gru = nn.GRU(
                prj_size,
                hidden_size,
                num_layers=num_heads,
                batch_first=True,
                dropout=drop,
            )
            self.linear = nn.Linear(hidden_size, prj_size)
            self.ln_gru = nn.LayerNorm(prj_size)
        elif aggragation == "self-cross-attention":
            self.self_att = nn.TransformerEncoderLayer(
                prj_size,
                num_heads,
                batch_first=True,
                activation=F.gelu,
                dim_feedforward=hidden_size,
                dropout=drop,
            )
            self.cross_att = nn.MultiheadAttention(
                prj_size, num_heads, batch_first=True
            )
            self.ln_cross_att = nn.LayerNorm(prj_size)
        elif aggragation == "learned_query":
            self.query = nn.Parameter(torch.randn(2, prj_size))
            self.attention = nn.MultiheadAttention(
                prj_size, num_heads, batch_first=True, dropout=drop
            )
            self.ln_att = nn.LayerNorm(prj_size)

        elif aggragation != "mean":
            raise ValueError(f"[MODEL] Invalid aggregation type: {aggragation}")

        print(f"[MODEL] Using aggregation type: {aggragation}")

        self.drop = nn.Dropout(drop)

        self.users = nn.Embedding(n_users, emb_size)

        self.linear1 = nn.Linear(emb_size, hidden_size)
        self.fc1 = nn.Linear(emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, prj_size)
        self.fc2 = nn.Linear(hidden_size, prj_size)
        self.ln_usr = nn.LayerNorm(prj_size)
        self.linear3 = nn.Linear(emb_size, prj_size)

        self.temp = temp
        if lt:
            print("[MODEL] Using learnable temperature")
            self.temp = nn.Parameter(torch.log(torch.tensor(1 / temp)))

        if self.prj_type == "linear":
            self.fc3 = nn.Linear(prj_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, prj_size)
            self.ln = nn.LayerNorm(prj_size)
        elif self.prj_type == "shared":
            self.f5 = nn.Linear(prj_size, hidden_size)
            self.f6 = nn.Linear(hidden_size, prj_size)
            self.ln2 = nn.LayerNorm(prj_size)
        elif self.prj_type == "shared+linear":
            self.f5 = nn.Linear(prj_size, hidden_size)
            self.f6 = nn.Linear(hidden_size, prj_size)
            self.fc3 = nn.Linear(prj_size, hidden_size)
            self.fc4 = nn.Linear(hidden_size, prj_size)
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
        self.drop(usr_x)
        usr_x = F.gelu(self.fc2(usr_x)) + self.linear2(usr_x) + self.linear3(user_embs)
        usr_x = self.ln_usr(usr_x)

        batch_size = music_embs.size(0)
        N_EMB = music_embs.size(1)

        if self.training and self.noise_level > 0:
            music_embs = music_embs + torch.randn_like(music_embs) * self.noise_level

        if self.aggragation == "mean":
            music_x = music_embs.mean(dim=2)
        elif self.aggragation == "weighted":
            music_embs = music_embs.permute(0, 1, 3, 2)
            music_x = (music_embs * self.weights).sum(dim=3)
        elif self.aggragation == "gating":
            # [batch, NPOS+NNEG, 13, emb_size] -> [batch, NPOS+NNEG, 13*emb_size]
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, self.prj_size * 13)
            # compute weights for each dimension of the music embeddings to be normalized
            weights = self.gate(music_embs_reshaped).view(
                batch_size, N_EMB, 13, self.prj_size
            )
            # normalize over the 13 dimensions
            weights = F.softmax(weights, dim=2)
            # apply weighted sum to the music embeddings
            music_embs = self.gate_norm(
                music_embs + F.gelu(self.gate_transform(music_embs))
            )
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.gate_out_norm(music_x)
        elif self.aggragation == "gating-tanh":
            # [batch, NPOS+NNEG, 13, emb_size] -> [batch, NPOS+NNEG, 13*emb_size]
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, self.prj_size * 13)
            # compute weights for each dimension of the music embeddings to be normalized
            weights = self.gate(music_embs_reshaped).view(
                batch_size, N_EMB, 13, self.prj_size
            )
            # normalize over the 13 dimensions
            weights = F.tanh(weights)
            # apply weighted sum to the music embeddings
            music_embs = self.gate_norm(
                music_embs + F.gelu(self.gate_transform(music_embs))
            )
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.gate_out_norm(music_x)
        elif self.aggragation == "gating-tanh":
            # [batch, NPOS+NNEG, 13, emb_size] -> [batch, NPOS+NNEG, 13*emb_size]
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, self.prj_size * 13)
            # compute weights for each dimension of the music embeddings to be normalized
            weights = self.gate(music_embs_reshaped).view(
                batch_size, N_EMB, 13, self.prj_size
            )
            # normalize over the 13 dimensions
            weights = F.tanh(weights)
            # apply weighted sum to the music embeddings
            music_embs = self.gate_norm(
                music_embs + F.gelu(self.gate_transform(music_embs))
            )
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.gate_out_norm(music_x)
        elif self.aggragation == "cross-attention":
            # use user embeddings as query and music embeddings as key and value
            # urs [batch, prj_size] -> [batch * NPOS+NNEG, 1, prj_size]
            # music [batch, NPOS+NNEG, 13, emb_size] -> [batch * NPOS+NNEG, 13, emb_size]
            user_queries = (
                usr_x.unsqueeze(1)
                .repeat(1, N_EMB, 1)
                .view(batch_size * N_EMB, 1, self.prj_size)
            )
            music_vals = music_embs.view(batch_size * N_EMB, 13, self.prj_size)
            music_x, weights = self.gate(user_queries, music_vals, music_vals)
            music_x = music_x.view(batch_size, N_EMB, self.prj_size) + music_embs.mean(
                dim=2
            )
            music_x = self.gate_norm(music_x)
        elif self.aggragation == "GRU":
            music_gru_input = music_embs.view(batch_size * N_EMB, 13, self.prj_size)
            music_x, _ = self.gru(music_gru_input)
            music_x = music_x[:, -1, :].view(
                batch_size, N_EMB, self.prj_size
            ) + music_embs.mean(dim=2)
            music_x = self.ln_gru(music_x)
        elif self.aggragation == "self-cross-attention":
            music_embs_input = music_embs.view(batch_size * N_EMB, 13, self.prj_size)
            music_self_att = self.self_att(music_embs_input)

            music_vals = music_self_att
            user_queries = (
                usr_x.unsqueeze(1)
                .repeat(1, N_EMB, 1)
                .view(batch_size * N_EMB, 1, self.prj_size)
            )
            music_x, _ = self.cross_att(user_queries, music_vals, music_vals)

            music_x = music_x.view(batch_size, N_EMB, self.prj_size)
            music_self_att = music_self_att.view(batch_size, N_EMB, 13, self.prj_size)
            music_x = music_x + music_self_att.mean(dim=2)
            music_x = self.ln_cross_att(music_x)
        elif self.aggragation == "learned_query":
            query = self.query.unsqueeze(0).repeat(batch_size * N_EMB, 1, 1)
            music_embs_input = music_embs.view(batch_size * N_EMB, 13, self.prj_size)
            music_x, _ = self.attention(query, music_embs_input, music_embs_input)
            music_x = music_x.view(batch_size, N_EMB, -1, self.prj_size).mean(
                dim=2
            ) + music_embs.mean(dim=2)
            self.ln_att(music_x)

        if self.prj_type == "linear":
            music_x = self.fc4(self.drop(F.gelu(self.fc3(music_x)))) + music_x
            music_x = self.ln(music_x)
        elif self.prj_type == "shared":
            music_x = self.f6(self.drop(F.gelu(self.f5(music_x)))) + music_x
            music_x = self.ln2(music_x)
            usr_x = self.f6(self.drop(F.gelu(self.f5(usr_x)))) + usr_x
            usr_x = self.ln2(usr_x)
        elif self.prj_type == "shared+linear":
            music_x = self.fc4(self.drop(F.gelu(self.fc3(music_x)))) + music_x
            music_x = self.f6(self.drop(F.gelu(self.f5(music_x)))) + music_x
            music_x = self.ln(music_x)
            usr_x = self.f6(self.drop(F.gelu(self.f5(usr_x)))) + usr_x
            usr_x = self.ln(usr_x)
        elif self.prj_type == "ln":
            music_x = self.ln(music_x)
        elif self.prj_type == "bn":
            music_x = music_x.permute(0, 2, 1)
            music_x = self.bn(music_x)
            music_x = music_x.permute(0, 2, 1)

        if self.training:
            drop_mask = (torch.rand(self.prj_size) < self.drop.p).to("cuda")
            music_x = music_x * drop_mask
            usr_x = usr_x * drop_mask

        return usr_x, music_x, self.temp

    def load_model(model_path, device="cuda"):
        model_savefile = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = model_savefile["model"]
        config = model_savefile["config"]
        opt_state = model_savefile["optimizer"]

        return state_dict, config, opt_state

    def calculate_score(self, user_embedding, music_embedding):
        """
        Calculate the score by computing the cosine similarity between the user and music embeddings.

        Args:
            user_embedding (torch.Tensor): The user embedding.
            music_embedding (torch.Tensor): The music embedding.

        Returns:
            torch.Tensor: The score.
        """

        out = user_embedding.unsqueeze(1)
        out = out.repeat(1, music_embedding.shape[1], 1)
        
        sim = self.cosine_similarity(out, music_embedding)

        sim = sim.view(-1, music_embedding.shape[1])

        return sim


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

        self.users = nn.Embedding(n_users, emb_size)

        self.fc1 = nn.Linear(emb_size, 2048)
        self.dropmid = nn.Dropout(drop)
        self.fc2 = nn.Linear(2048, prj_size)

        self.temp = temp
        if lt:
            print("[MODEL] Using learnable temperature")
            self.temp = nn.Parameter(torch.tensor(self.temp))

        if self.prj_type == "linear":
            self.ln2 = nn.LayerNorm(prj_size)
            self.fc3 = nn.Linear(prj_size, 2048)
            self.fc4 = nn.Linear(2048, prj_size)
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

        music_embs = music_embs.mean(dim=2)
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

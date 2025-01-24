#!/usr/bin/env python3
"""
visualize_user_embeddings.py

A script to visualize user embeddings from the AlignerV2 model using t-SNE or UMAP.

Usage:
    python visualize_user_embeddings.py --model_path "path/to/AlignerV2_best_model.pt" \
                                       --method "tsne" \
                                       --output_path "user_embeddings.png" \
                                       --device "cuda" \
                                       --plot_type "2d"

Arguments:
    --model_path: Path to the AlignerV2 model checkpoint.
    --method: Dimensionality reduction method ('tsne' or 'umap').
    --output_path: File path to save the visualization image.
    --device: Computation device ('cuda' or 'cpu').
    --plot_type: Type of plot ('2d' or '3d').

Example:
    python visualize_user_embeddings.py --model_path "checkpoints/AlignerV2_best_model.pt" \
                                       --method "umap" \
                                       --output_path "user_embeddings_umap.png" \
                                       --device "cuda" \
                                       --plot_type "2d"
"""

import argparse
import os
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from umap import UMAP

# --- Define the AlignerV2 class ---
# Ensure that this class definition is identical to the one used during training.
# If AlignerV2 is defined in a separate module, consider importing it instead.
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
            self.gate = nn.Linear(prj_size * 13, prj_size * 13)
            self.gate_transform = nn.Linear(prj_size, prj_size)
            self.gate_norm = nn.LayerNorm(prj_size)
            self.gate_out_norm = nn.LayerNorm(prj_size)
        elif aggragation == "gating-tanh":
            self.gate = nn.Linear(prj_size * 13, prj_size * 13)
            self.gate_transform = nn.Linear(prj_size, prj_size)
            self.gate_norm = nn.LayerNorm(prj_size)
            self.gate_out_norm = nn.LayerNorm(prj_size)
        elif aggragation == "cross-attention":
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
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, self.prj_size * 13)
            weights = self.gate(music_embs_reshaped).view(
                batch_size, N_EMB, 13, self.prj_size
            )
            weights = F.softmax(weights, dim=2)
            music_embs = self.gate_norm(
                music_embs + F.gelu(self.gate_transform(music_embs))
            )
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.gate_out_norm(music_x)
        elif self.aggragation == "gating-tanh":
            music_embs_reshaped = music_embs.view(batch_size, N_EMB, self.prj_size * 13)
            weights = self.gate(music_embs_reshaped).view(
                batch_size, N_EMB, 13, self.prj_size
            )
            weights = F.tanh(weights)
            music_embs = self.gate_norm(
                music_embs + F.gelu(self.gate_transform(music_embs))
            )
            music_x = (weights * music_embs).sum(dim=2)
            music_x = self.gate_out_norm(music_x)
        elif self.aggragation == "cross-attention":
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
            drop_mask = (torch.rand(self.prj_size) < self.drop.p).to(music_embs.device)
            music_x = music_x * drop_mask
            usr_x = usr_x * drop_mask

        return usr_x, music_x, self.temp

    @staticmethod
    def load_model(model_path, device="cuda"):
        model_savefile = torch.load(model_path, map_location=device)
        state_dict = model_savefile["model"]
        config = model_savefile["config"]
        opt_state = model_savefile.get("optimizer", None)

        return state_dict, config, opt_state


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize user embeddings from AlignerV2 model using t-SNE or UMAP."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the AlignerV2 model checkpoint (e.g., 'checkpoints/AlignerV2_best_model.pt').",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["tsne", "umap", "pca"],
        default="tsne",
        help="Dimensionality reduction method to use: 'tsne' or 'umap'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Computation device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--plot_type",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Type of plot: '2d' or '3d'.",
    )
    parser.add_argument(
        "--no_labels",
        action="store_true",
        help="If set, user labels will not be displayed on the plot.",
    )
    return parser.parse_args()


def load_aligner_model(model_path, device):
    """
    Loads the AlignerV2 model from the given checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        device (str): Computation device ('cuda' or 'cpu').

    Returns:
        AlignerV2: Loaded AlignerV2 model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'.")

    print(f"[INFO] Loading model from '{model_path}' on device '{device}'...")
    state_dict, config, _ = AlignerV2.load_model(model_path, device=device)

    # Extract necessary configuration parameters
    n_users = config.get("nusers", None)
    emb_size = config.get("emb_size", None)
    prj_size = config.get("prj_size", 512)
    prj_type = config.get("prj", "linear")
    aggr = config.get("aggr", "mean")
    noise_level = config.get("noise_level", 0.0)
    lt = config.get("learnable_temp", False)
    temp = config.get("temp", 0.07)
    drop = config.get("drop", 0.35)
    
    if n_users is None or emb_size is None:
        raise ValueError("[ERROR] 'nusers' and 'emb_size' must be specified in the config.")

    # Initialize the model
    aligner_model = AlignerV2(
        n_users=n_users,
        emb_size=emb_size,
        prj_size=prj_size,
        prj_type=prj_type,
        aggragation=aggr,
        noise_level=noise_level,
        lt=lt,
        temp=temp,
        drop=drop,
    ).to(device)

    # Load state dict
    aligner_model.load_state_dict(state_dict)
    aligner_model.eval()
    print("[INFO] Model loaded and set to evaluation mode.")

    return aligner_model


def reduce_dimensions(embeddings, method="tsne", n_components=2, random_state=42):
    """
    Reduces the dimensionality of the embeddings using t-SNE or UMAP.

    Args:
        embeddings (np.ndarray): The embeddings to reduce, shape (n_samples, emb_dim).
        method (str): 'tsne' or 'umap'.
        n_components (int): Target number of dimensions (2 or 3).
        random_state (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Reduced embeddings, shape (n_samples, n_components).
    """
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, random_state=random_state, init='random')
        reduced = reducer.fit_transform(embeddings)
        print(f"[INFO] Reduced dimensions using t-SNE to shape {reduced.shape}.")
    elif method.lower() == "umap":
        reducer = UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        print(f"[INFO] Reduced dimensions using UMAP to shape {reduced.shape}.")
    elif method.lower() == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
        print(f"[INFO] Reduced dimensions using PCA to shape {reduced.shape}.")
    else:
        raise ValueError("Invalid method. Choose either 'tsne' or 'umap'.")
    return reduced


def plot_embeddings(
    reduced_embeddings,
    user_ids,
    method="tsne",
    plot_type="2d",
    no_labels=False,
):
    """
    Plots the reduced embeddings.

    Args:
        reduced_embeddings (np.ndarray): Embeddings after dimensionality reduction, shape (n_users, 2 or 3).
        user_ids (list or np.ndarray): List of user IDs corresponding to each embedding.
        method (str): 'tsne' or 'umap'.
        plot_type (str): '2d' or '3d'.
        output_path (str): Path to save the plot image.
        no_labels (bool): If True, do not display user labels on the plot.
    """
    n_users = reduced_embeddings.shape[0]
    plt.figure(figsize=(10, 8))
    if plot_type == "3d":
        ax = plt.axes(projection='3d')
        scatter = ax.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            reduced_embeddings[:, 2],
            c=user_ids,
            cmap='tab20',
            s=60,
            alpha=0.8,
            edgecolors='w'
        )
        ax.set_title(f"User Embeddings Visualization ({method.upper()} 3D)", fontsize=15)
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_zlabel("Dimension 3", fontsize=12)
    else:
        scatter = plt.scatter(
            reduced_embeddings[:, 0],
            reduced_embeddings[:, 1],
            c=user_ids,
            cmap='tab20',
            s=60,
            alpha=0.8,
            edgecolors='w'
        )
        plt.title(f"User Embeddings Visualization ({method.upper()} 2D)", fontsize=15)
        plt.xlabel("Dimension 1", fontsize=12)
        plt.ylabel("Dimension 2", fontsize=12)

    plt.colorbar(scatter, label='User ID', ticks=range(n_users))
    
    if not no_labels:
        for i in range(n_users):
            if plot_type == "3d":
                ax.text(
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1],
                    reduced_embeddings[i, 2],
                    f"{user_ids[i]}",
                    fontsize=8,
                    ha='center',
                    va='center'
                )
            else:
                plt.text(
                    reduced_embeddings[i, 0],
                    reduced_embeddings[i, 1],
                    f"{user_ids[i]}",
                    fontsize=8,
                    ha='center',
                    va='center'
                )

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    args = parse_arguments()

    # Validate plot_type
    if args.plot_type not in ["2d", "3d"]:
        raise ValueError("plot_type must be either '2d' or '3d'.")

    # Load the AlignerV2 model
    aligner_model = load_aligner_model(args.model_path, args.device)

    # Extract user embeddings
    with torch.no_grad():
        user_embeddings = aligner_model.users.weight.cpu().numpy()
    user_ids = np.arange(user_embeddings.shape[0])

    # Reduce dimensions
    n_components = 3 if args.plot_type == "3d" else 2
    reduced_embeddings = reduce_dimensions(
        user_embeddings,
        method=args.method,
        n_components=n_components,
        random_state=42
    )

    # Plot embeddings
    plot_embeddings(
        reduced_embeddings,
        user_ids=user_ids,
        method=args.method,
        plot_type=args.plot_type,
        no_labels=args.no_labels,
    )


if __name__ == "__main__":
    main()

from enum import Enum

import torch
import torch.nn as nn


class Categories(Enum):
    listen = 0
    skip = 1
    like = 2
    dislike = 3
    loved = 4
    hated = 5


def split_pos_neg_music_embedding(music_x: torch.Tensor):
    """
    Split the positive and negative music embeddings from the input tensor

    Args:
        music_x (torch.Tensor): The input tensor

    Returns:
        torch.Tensor: The positive music embedding
        torch.Tensor: The negative music embedding
    """
    posemb_out = music_x[:, 0, :].unsqueeze(dim=1)
    negemb_out = music_x[:, 1:, :]

    return posemb_out, negemb_out


def weighted_contrastive_loss(usr_emb, posemb, negemb, weights, loss_weight, temp=0.07):
    cos = nn.CosineSimilarity(dim=2, eps=1e-6)
    possim = cos(out, posemb)

    out = out.repeat(1, negemb.shape[1], 1)
    negsim = cos(out, negemb)

    # breakpoint()
    logits = torch.cat((possim, negsim), dim=1) * torch.exp(temp)
    exp = torch.exp(logits)
    denom = torch.sum(exp, dim=1) + 1e-6
    loss = -torch.log(exp[:, 0] / denom)

    loss = loss * ((weights * loss_weight) + 1)
    loss = torch.mean(loss)
    return loss

class ScoreToFeedback():
    def __init__(self, device):
        num_categories = 3

        # Dislike (-1): [-1, -0.33)
        # Neutral (0): [-0.33, 0.33)
        # Like (1): [0.33, 1]

        self.bin_edges = torch.linspace(-1, 1, num_categories + 1).to(device)
        
    def get_feedback(self, scores):
        """
        
        """
        # Map `target_score` to bins [0, num_categories-1]
        target_labels = torch.bucketize(scores, self.bin_edges, right=True) - 1

        return target_labels


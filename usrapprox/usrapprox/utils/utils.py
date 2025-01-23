from enum import Enum

import torch


class ScoreToFeedback:
    def __init__(self, device, num_categories: int = 2):
        # Dislike (-1): [-1, 0)
        # Like (1): [0, 1]
        self.num_categories = num_categories
        if num_categories == 2:  # Two categories: Dislike (-1) and Like (1)
            self.bin_edges = torch.linspace(-1, 1, num_categories + 1).to(device)
        elif num_categories == 3:
            self.bin_edges = torch.linspace(-1, 1, num_categories + 1).to(device)
        else:
            raise ValueError(
                f"num_categories should be 2 or 3, but got {num_categories}"
            )
        self.device = device

    def get_feedback(self, scores):
        """
        Maps scores to feedback values of -1 or 1.
        """
        # Map scores to bins: [-1, 0) -> -1, [0, 1] -> 1
        if self.num_categories == 2:
            target_labels = torch.bucketize(scores, self.bin_edges, right=True)
            target_labels = 2 * target_labels - 3  # Convert to -1 or 1
        else:
            target_labels = torch.bucketize(scores, self.bin_edges, right=True) - 2

        return target_labels.to(self.device)

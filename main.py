import json
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

from usrapprox.usrapprox.models.aligner_v2 import AlignerWrapper
from usrapprox.usrapprox.models.probabilistic import (
    calculate_logits,
    probabilistic_model_torch,
)
from usrapprox.usrapprox.models.usr_emb import UsrEmb

# from usrapprox.usrapprox.utils.utils import Categories
from usrapprox.usrapprox.utils.config import AlignerV2Config
from usrapprox.usrapprox.utils.utils import Categories
from usrembeds.datautils.dataset import ContrDatasetMERT
from usrembeds.models.model import AlignerV2

# from usrapprox.models.probabilistic import calculate_logits, probabilistic_model_torch
# from usrapprox.utils.utils import Categories

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!
print("DEVICE: ", DEVICE)

if __name__ == "__main__":
    alignerv2 = AlignerWrapper(device=DEVICE)
    user_embedder = UsrEmb(device=DEVICE)

    # DONE: test if alignerv2wrapper works with forward methods
    # DONE: why the prob model is never 1?
    # TODO: add score column to UsrEmb
    # TODO: modify UsrEmb to have a forward method that takes also the score
    # TODO: test UsrEmb and it's forward
    # TODO: simple loss fn based on Lollo's
    # TODO: try to train UsrEmb with the same data as AlignerV2

    membs_path = "usrembeds/data/embeddings/embeddings_full_split"
    stats_path = "clean_stats.csv"
    splits_path = "usrembeds/data/splits.json"

    with open(splits_path, "r") as f:
        splits = json.load(f)

    test = splits["test"]
    users = splits["users"]

    dataset = ContrDatasetMERT(
        membs_path,
        stats_path,
        split=test,
        usrs=users,
        nneg=AlignerV2Config.neg_samples,
        multiplier=AlignerV2Config.multiplier,
        transform=None,
    )

    _, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
    )

    counter = 0
    for tracks in tqdm(val_dataloader):
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        music_embedding = torch.cat((posemb, negemb), dim=1)

        user_embedding, pos_feedback_wrt_song, neg_feedback_wrt_song = alignerv2(idx, music_embedding)

        print(f"Pos embedding shape: {posemb.shape}") # torch.Size([16, 1, 13, 768])
        print(f"Neg embedding shape: {negemb.shape}") # torch.Size([16, 20, 13, 768])

        print(f"Pos feedback shape: {pos_feedback_wrt_song.shape}") # torch.Size([16, 1])
        print(f"Neg feedback shape: {neg_feedback_wrt_song.shape}") # torch.Size([16, 20])

        music_feedback = torch.cat((pos_feedback_wrt_song, neg_feedback_wrt_song), dim=1)

        print("--------------------------------------")
        print(f"Music embedding shape: {music_embedding.shape}") # torch.Size([16, 21, 13, 768])
        print(f"Music feedback shape: {music_feedback.shape}") # torch.Size([16, 21])

        print("--------------------------------------")
        
        # Add dimensions to `music_feedback` to match `music_embedding`
        music_feedback_expanded = music_feedback.unsqueeze(-1).unsqueeze(-1)  # Shape: [16, 21, 1, 1]
        print(f"Music feedback expanded shape: {music_feedback_expanded.shape}")

        # Expand the feedback dimensions to match `music_embedding`
        music_feedback_expanded = music_feedback_expanded.expand(-1, -1, 13, 1)  # Shape: [16, 21, 13, 768]
        print(f"Music feedback expanded shape: {music_feedback_expanded.shape}")

        # Concatenate along the last dimension
        music_combined = torch.cat((music_embedding, music_feedback_expanded), dim=-1)  # Shape: [16, 21, 13, 769]

        print(f"Music combined shape: {music_combined.shape}")

        print("--------------------------------------")
        print(music_feedback[0][1])
        print(music_combined[0][1][0][768])

        exit()

        # # print(user_embedding.shape, pos_feedback_wrt_song.shape, neg_feedback_wrt_song.shape)
        # music_feedback = torch.cat((pos_feedback_wrt_song, neg_feedback_wrt_song), dim=0)
        # print(music_embedding.shape)
        # print(music_feedback.shape)
        # exit()
        # user_embedder()

        # print(usr_feedback_wrt_song)
        # break

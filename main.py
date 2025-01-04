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

        user_embedding, music_feedback = alignerv2(idx, music_embedding)
        print(f"User embedding shape: {user_embedding.shape}")
        print(f"Music feedback shape: {music_feedback.shape}")
        # Music combined shape: [16, 21, 13, 769]
        music_combined = torch.cat((music_embedding, music_feedback), dim=-1)  

        usr_x , music_x = user_embedder(music_combined)

        print(f"User embedding shape: {usr_x.shape}")
        print(f"Music embedding shape: {music_x.shape}")





        break

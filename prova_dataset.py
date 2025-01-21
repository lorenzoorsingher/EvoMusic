import os
from tqdm import tqdm
import torch
import json

from usrapprox.usrapprox.models.aligner_v2 import AlignerV2Wrapper
from usrapprox.usrapprox.utils.dataset import UserDefinedContrastiveDataset

# set seed and torch deterministic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_id = 0

    alignerV2 = AlignerV2Wrapper()

    dataset = UserDefinedContrastiveDataset(
        alignerV2,
        splits_path="usrembeds/data/splits.json",
        embs_path="usrembeds/data/embeddings/embeddings_full_split",
        npos=1,
        nneg=1,
        batch_size=5,
        num_workers=10,
        user_id=user_id,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=10
    )

    for batch in tqdm(dataloader):
        positives = batch["positives"]
        negatives = batch["negatives"]
        pos_scores = batch["pos_scores"]
        neg_scores = batch["neg_scores"]

        full_batch = torch.cat((positives, negatives), dim=1)
        full_scores = torch.cat((pos_scores, neg_scores), dim=1)

        index_tensor = torch.LongTensor([user_id] * full_batch.shape[0]).to(device)
        full_batch = full_batch.to(device)

        _, _, result = alignerV2(index_tensor, full_batch)

        result = result.to("cpu")
        full_scores = full_scores.to("cpu")

        print(f"predicted score: {result}")
        print(f"real score: {full_scores}")

        print("------------------------")

        _, _, result = alignerV2(index_tensor, full_batch)

        result = result.to("cpu")
        full_scores = full_scores.to("cpu")

        print(f"predicted score: {result}")
        print(f"real score: {full_scores}")

        assert torch.allclose(result, full_scores), "Result does not match scores!"
        print("Test passed!")
        break

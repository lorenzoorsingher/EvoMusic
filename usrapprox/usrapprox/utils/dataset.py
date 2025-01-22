import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from usrapprox.usrapprox.models.aligner_v2 import AlignerV2Wrapper


class AllSongsDataset(Dataset):
    def __init__(self, splits_path, embs_path, partition="train"):
        self.embs_path = embs_path
        with open(splits_path, "r") as f:
            splits = json.load(f)
        self.splits = splits[partition]

    def __getitem__(self, index):
        # Return pairs of embeddings (index and index+1)
        embedding1 = self.__get_embedding(index)
        embedding2 = self.__get_embedding(index)  # Ensure valid index
        return torch.Tensor([embedding1, embedding2]), index

    def __len__(self):
        # return len(self.splits)
        return 300

    def __get_embedding(self, idx):
        song_id = self.splits[idx]
        emb_file = os.path.join(self.embs_path, f"{song_id}.json")
        if os.path.isfile(emb_file):
            try:
                with open(emb_file, "r") as f:
                    data = json.load(f)
                    if song_id in data:
                        return data[song_id][0]
                    else:
                        print("No embeddings for song_id")
                        return [0.0]
            except:
                print("Error reading file")
                return [0.0]
        else:
            print("File does not exist")
            return [0.0]


class UserDefinedContrastiveDataset(Dataset):
    def __init__(
        self,
        alignerV2: AlignerV2Wrapper,
        splits_path,
        embs_path,
        user_id=0,
        npos=1,
        nneg=1,
        batch_size=128,
        num_workers=10,
        partition="train",
    ):
        self.embs_path = embs_path
        with open(splits_path, "r") as f:
            splits = json.load(f)
        self.splits = splits[partition]
        self.index_to_song_id = {
            idx: song_id for idx, song_id in enumerate(self.splits)
        }

        all_songs_dataset = AllSongsDataset(splits_path, embs_path, partition)
        dataloader = DataLoader(
            all_songs_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alignerV2.to(device)

        self.positive_samples = []
        self.negative_samples = []

        # Process feedback for song pairs
        for emb, indices in tqdm(dataloader, desc="Processing Feedback"):
            emb = emb.to(device)
            index_tensor = torch.LongTensor([user_id] * emb.shape[0]).to(device)

            # batch = torch.cat((emb1, emb2), dim=1)
            batch = emb
            _, _, _, feedback_scores = alignerV2(index_tensor, batch)

            feedback_scores = feedback_scores.cpu().tolist()
            for idx, score in zip(indices.tolist(), feedback_scores):
                song_id = self.index_to_song_id[idx]
                # for score in score_vector:
                if score[0] > 0:
                    self.positive_samples.append((song_id, score[0]))
                else:
                    self.negative_samples.append((song_id, score[0]))

        self.npos = npos
        self.nneg = nneg

    def __getitem__(self, index):
        assert len(self.positive_samples) >= self.npos, "Not enough positive samples."
        assert len(self.negative_samples) >= self.nneg, "Not enough negative samples."

        pos_samples = torch.randperm(len(self.positive_samples))[: self.npos]
        neg_samples = torch.randperm(len(self.negative_samples))[: self.nneg]

        positives = [
            self.__get_embedding(self.positive_samples[i][0]) for i in pos_samples
        ]
        negatives = [
            self.__get_embedding(self.negative_samples[i][0]) for i in neg_samples
        ]

        # pos_scores = [self.positive_samples[i][1] for i in pos_samples]
        # neg_scores = [self.negative_samples[i][1] for i in neg_samples]

        # pos = self.positive_samples[index]
        # positives = [self.__get_embedding(pos[0])]
        # pos_scores = [pos[1]]

        # neg = self.negative_samples[index]
        # negatives = [self.__get_embedding(neg[0])]
        # neg_scores = [neg[1]]

        return torch.Tensor(positives), torch.Tensor(negatives)


    def __len__(self):
        # return len(self.positive_samples) #+ len(self.negative_samples)
        return 300


    def __get_embedding(self, song_id):
        emb_file = os.path.join(self.embs_path, f"{song_id}.json")
        if os.path.isfile(emb_file):
            try:
                with open(emb_file, "r") as f:
                    data = json.load(f)
                    if song_id in data:
                        return data[song_id][0]
                    else:
                        print("No embeddings for song_id")
                        return [0.0]
            except:
                print("Error reading file")
                return [0.0]
        else:
            print("File does not exist")
            return [0.0]

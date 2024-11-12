import json
import os
import torch
import pandas as pd
import librosa
import numpy as np
import scipy.signal as sps


from torch.utils.data import Dataset
from tqdm import tqdm
from random import randint


class ContrDataset(Dataset):
    def __init__(
        self,
        embs_dir,
        stats_dir,
        nneg=10,
        multiplier=10,
        transform=None,
    ):
        self.embs_dir = embs_dir
        self.stats_dir = stats_dir
        self.nneg = nneg
        self.multiplier = multiplier
        self.transform = transform

        print("[DATASET] Loading files and keys")
        embedding_files = [
            f for f in os.listdir(embs_dir) if os.path.isfile(os.path.join(embs_dir, f))
        ]
        embedding_files.remove("allkeys.json")

        with open(os.path.join(embs_dir, "allkeys.json"), "r") as f:
            self.allkeys = json.load(f)

        self.allkeys.remove("metadata")

        # del self.allkeys["metadata"]
        # mapping the keys to a list because dict lookup is just too slow
        self.emb_map = {key: idx for idx, key in enumerate(self.allkeys)}

        self.emb_list = [[] for _ in self.allkeys]

        print("[DATASET] Loading embeddings")
        for num, file in enumerate(tqdm(embedding_files)):
            with open(os.path.join(embs_dir, file), "r") as f:
                data = json.load(f)
                for key, value in data.items():
                    if key != "metadata":
                        # print(len(value[0]))
                        # if len(value[0]) != 512:
                        #     breakpoint()
                        self.emb_list[self.emb_map[key]].extend(value)

        print("[DATASET] Loading users stats")
        # load the stats
        self.stats = pd.read_csv(stats_dir)
        self.stats["count"] = self.stats["count"].astype(int)

        # remove tracks with no embeddings
        self.stats = self.stats[self.stats["id"].isin(self.allkeys)].reset_index(
            drop=True
        )

        self.idx2usr = self.stats["userid"].unique().tolist()

        # Group by 'userid' and aggregate 'id' into a list
        # self.user2songs = self.stats.groupby("userid")["id"].apply(list).to_dict()

        # breakpoint()

        self.usersums = self.stats.groupby("userid")["count"].sum()
        self.userstd = self.stats.groupby("userid")["count"].std()
        self.usercount = self.stats.groupby("userid")["count"].count()

        self.user2songs = (
            self.stats.groupby("userid")
            .apply(lambda x: list(zip(x["id"], x["count"])))
            .to_dict()
        )
        # breakpoint()
        # number of users
        self.nusers = self.stats["userid"].nunique()

        # breakpoint()

    def __len__(self):
        return self.nusers * self.multiplier

    def __getitem__(self, idx):

        idx = idx % self.nusers
        # TODO: implement way to use playcount to select positive samples
        usr = self.idx2usr[idx]
        # breakpoint()
        pos = self.user2songs[usr]
        # count = torch.Tensor(count).type(torch.int32)
        neg = list(set(self.allkeys) - set(pos))

        pos_sample = pos[randint(0, len(pos) - 1)]
        posset, count = pos_sample

        mean = self.usersums[usr] / self.usercount[usr]
        top70 = mean + self.userstd[usr]
        weight = min(1, count / top70)
        # weight = torch.Tensor([weight])
        # breakpoint()

        negset = np.random.choice(neg, size=self.nneg, replace=False)

        poslist = []
        # for pos in posset:
        #     embs = self.emb_list[self.emb_map[pos]]
        #     # if len(embs) == 0:
        #     #     print(f"Empty embedding for {pos}")
        #     #     breakpoint()
        #     poslist.append(embs[randint(0, len(embs) - 1)])

        embs = self.emb_list[self.emb_map[posset]]
        poslist.append(embs[randint(0, len(embs) - 1)])

        neglist = []
        for neg in negset:
            embs = self.emb_list[self.emb_map[neg]]
            # if len(embs) == 0:
            #     print(f"Empty embedding for {neg}")
            #     breakpoint()
            neglist.append(embs[randint(0, len(embs) - 1)])

        # neglist = [
        #     self.emb_list[self.emb_map[neg]][
        #         randint(0, len(self.emb_list[self.emb_map[neg]]))
        #     ]
        #     for neg in negset
        # ]

        posemb = torch.Tensor(poslist)
        negemb = torch.Tensor(neglist)

        # print(negemb.shape)
        # breakpoint()
        return idx, posemb, negemb, weight


class MusicDataset(Dataset):
    def __init__(
        self,
        data_dir,
        type="audio",
        audio_len=1,
        resample=None,
        nsamples=None,
        repeat=1,
        transform=None,
    ):
        self.data_dir = data_dir
        self.type = type
        self.audio_len = audio_len
        self.resample = resample
        self.repeat = repeat

        self.transform = transform

        # load the songs
        self.tracks_paths = [
            os.path.join(data_dir, track)
            for track in os.listdir(data_dir)
            if track.endswith(".mp3")
        ]

        if nsamples is not None:
            self.tracks_paths = self.tracks_paths[:nsamples]

    def __len__(self):
        return len(self.tracks_paths) * self.repeat

    def __getitem__(self, idx):

        idx = idx % len(self.tracks_paths)

        stat = {}
        track_path = self.tracks_paths[idx]
        stat["id"] = track_path.split("/")[-1].split("_")[0]

        # Convert mp3 to wav
        y, sr = librosa.load(track_path, sr=None)
        stat["sr"] = sr
        if self.resample is not None:
            # Resample data
            number_of_samples = round(len(y) * float(self.resample) / sr)
            y = sps.resample(y, number_of_samples)
            stat["sr"] = self.resample

        # if audio is too short, repeat it
        pad_times = int((sr * self.audio_len) / len(y))
        y = np.tile(y, pad_times + 1)

        # take random audio_len sec long snippet
        snip_len = min(len(y), sr * self.audio_len) - 1
        snip_idx = np.random.randint(0, len(y) - snip_len)
        snip = y[snip_idx : snip_idx + snip_len]

        return stat, snip


if __name__ == "__main__":

    music_path = "../scraper/music"
    membs_path = "usrembeds/data/embeddings/batched"
    stats_path = "clean_stats.csv"

    dataset = ContrDataset(membs_path, stats_path, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for track in tqdm(dataloader):
        idx, posemb, negemb, weights = track
        breakpoint()

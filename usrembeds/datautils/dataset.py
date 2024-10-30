import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import librosa
import numpy as np

from scipy.io import wavfile
import scipy.signal as sps
from io import BytesIO


class MusicDataset(Dataset):
    def __init__(
        self,
        data_dir,
        stats_dir,
        type="audio",
        audio_len=1,
        resample=None,
        mel_cfg=None,
        transform=None,
    ):
        self.data_dir = data_dir
        self.stats_dir = stats_dir
        self.type = type
        self.audio_len = audio_len
        self.resample = resample

        if mel_cfg is not None:
            self.hop_len = mel_cfg["hop_length"]
            self.n_mels = mel_cfg["n_mels"]

        self.transform = transform

        # load the songs
        tracks_paths = [
            os.path.join(data_dir, track)
            for track in os.listdir(data_dir)
            if track.endswith(".mp3")
        ]

        self.tracks = {
            track.split("/")[-1].split("_")[0]: track for track in tracks_paths
        }

        # load the stats
        self.stats = pd.read_csv(stats_dir)
        self.stats["count"] = self.stats["count"].astype(int)

        # create a mapping
        self.mapping = []
        counts = self.stats["count"].to_list()
        for idx, num in enumerate(counts):
            self.mapping.extend([idx] * num)

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):

        stat = self.stats.iloc[self.mapping[idx]]
        stat = stat.to_dict()
        track_id = stat["id"]
        track_path = self.tracks[track_id]

        # Convert mp3 to wav
        y, sr = librosa.load(track_path, sr=None)
        stat["sr"] = sr
        if self.resample is not None:
            # Resample data
            number_of_samples = round(len(y) * float(self.resample) / sr)
            y = sps.resample(y, number_of_samples)
            stat["sr"] = self.resample

        # print("loaded mp3")
        # take random audio_len long sec snippet
        snip_len = min(len(y), sr * self.audio_len) - 1
        snip_idx = np.random.randint(0, len(y) - snip_len)
        snip = y[snip_idx : snip_idx + snip_len]

        if self.type == "mel":
            # Compute the Mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(
                y=snip,
                sr=sr,
                n_mels=128,
                hop_length=int(self.hop_len * sr),
            )

            # Convert to log scale (decibels)
            data = librosa.power_to_db(mel_spectrogram, ref=np.max)
        elif self.type == "audio":
            data = snip

        # breakpoint()
        # if self.transform:
        #     image = self.transform(image)

        return stat, data


if __name__ == "__main__":

    music_path = "/home/lollo/Documents/python/BIO/project/scraper/music"
    stats_path = "/home/lollo/Documents/python/BIO/project/scraper/data/clean_stats.csv"

    dataset = MusicDataset(music_path, stats_path, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for track in dataloader:
        stat, mel = track
        breakpoint()

import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset
from utils import plot_music_batch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!

# torchopenl3.ge

# model = torchopenl3.OpenL3Embedding(
#     input_repr="mel128", embedding_size=512, content_type="music"
# )


if __name__ == "__main__":

    music_path = "/home/lollo/Documents/python/BIO/project/scraper/music"
    stats_path = "/home/lollo/Documents/python/BIO/project/scraper/data/clean_stats.csv"

    BATCH_SIZE = 16
    EMB_SIZE = 512
    HOP_SIZE = 0.2
    AUDIO_LEN = 5

    # mel_cfg = {
    #     "n_mels": 128,
    #     "hop_length": HOP_SIZE,
    #     "audio_len": AUDIO_LEN,
    # }

    dataset = MusicDataset(
        music_path,
        stats_path,
        type="audio",
        audio_len=AUDIO_LEN,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    for track in dataloader:
        stat, audio = track

        sr = stat["sr"][0].item()

        emb, ts = torchopenl3.get_audio_embedding(
            audio,
            sr,
            hop_size=HOP_SIZE,
            embedding_size=EMB_SIZE,
        )

        print(emb.shape)

        plot_music_batch(emb, DEVICE)

        breakpoint()

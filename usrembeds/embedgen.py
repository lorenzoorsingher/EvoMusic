import torchopenl3
import torch
import numpy as np

from datautils.dataset import MusicDataset
from utils import plot_music_batch
import time
import json
import os


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


if __name__ == "__main__":

    music_path = "../scraper/music"
    stats_path = "../scraper/data/clean_stats.csv"
    emb_path = "usrembeds/data/embeddings"

    SAVE_RATE = 500

    BATCH_SIZE = 32
    EMB_SIZE = 512

    HOP_SIZE = 0.1
    AUDIO_LEN = 3
    TARGET_SR = torchopenl3.core.TARGET_SR

    dataset = MusicDataset(
        music_path,
        stats_path,
        audio_len=AUDIO_LEN,
        resample=TARGET_SR,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = torchopenl3.core.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=EMB_SIZE,
    )

    emb_dict = {
        "metadata": {"hop_size": HOP_SIZE, "audio_len": AUDIO_LEN, "sr": TARGET_SR}
    }

    for idx, track in enumerate(dataloader):
        if idx >= 1224:
            start_time = time.time()

            stat, audio = track

            sr = stat["sr"][0].item()

            audio = audio.to(DEVICE)
            emb, ts = torchopenl3.get_audio_embedding(
                audio,
                sr,
                model=model,
                hop_size=HOP_SIZE,
                embedding_size=EMB_SIZE,
            )

            # print(emb.shape)
            # plot_music_batch(emb, DEVICE)
            mean_emb = emb.mean(axis=1)

            for i, track_id in enumerate(stat["id"]):
                if track_id not in emb_dict:
                    emb_dict[track_id] = []
                emb_dict[track_id].append(mean_emb[i].cpu().detach().tolist())

            if idx % SAVE_RATE == 0:
                with open(os.path.join(emb_path, f"embeddings_{idx}.json"), "w") as f:
                    json.dump(emb_dict, f)
            end_time = time.time()
            # print(f"{(end_time - start_time)/BATCH_SIZE} seconds per track")

        print(f"{idx}/{len(dataloader)}")

    with open(os.path.join(emb_path, f"embeddings_last.json"), "w") as f:
        json.dump(emb_dict, f)

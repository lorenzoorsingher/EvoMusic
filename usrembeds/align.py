import librosa
import torch
import torchopenl3
import numpy as np

import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.signal as sps
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from tqdm import tqdm

from datautils.dataset import ContrDataset
from models.model import Aligner


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # use GPU if we can!


def load_wav(track_path, resample, audio_len):

    # Convert mp3 to wav
    y, sr = librosa.load(track_path, sr=None)

    number_of_samples = round(len(y) * float(resample) / sr)
    y = sps.resample(y, number_of_samples)

    # if audio is too short, repeat it
    pad_times = int((sr * audio_len) / len(y))
    y = np.tile(y, pad_times + 1)

    # take random audio_len sec long snippet
    snip_len = min(len(y), sr * audio_len) - 1
    snip_idx = np.random.randint(0, len(y) - snip_len)
    snip = y[snip_idx : snip_idx + snip_len]

    return snip


if __name__ == "__main__":

    # load model and config from checkpoint
    LOAD = "usrembeds/checkpoints/run_20241107_201542_best.pt"
    model_state, config, _ = Aligner.load_model(LOAD)

    EMB_SIZE = config["emb_size"]
    MUSIC_EMB_SIZE = config["prj_size"]
    TEMP = config["temp"]
    LT = config["learnable_temp"]
    PRJ = config["prj"]
    NUSERS = config["nusers"]

    # load aligner model
    align_model = Aligner(
        n_users=NUSERS,
        emb_size=EMB_SIZE,
        prj_size=MUSIC_EMB_SIZE,
        prj_type=PRJ,
        lt=LT,
        temp=TEMP,
    ).to(DEVICE)

    align_model.load_state_dict(model_state)
    align_model.eval()

    # audio extraction setting
    HOP_SIZE = 0.1  # hop size defined in the paper
    TARGET_SR = torchopenl3.core.TARGET_SR
    AUDIO_LEN = 3

    # load embedder model
    embed_model = torchopenl3.core.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=MUSIC_EMB_SIZE,
    )

    # load wav file
    track_path = "/home/lollo/Documents/python/BIO/project/scraper/music/7zYoeHLPNU3X7AI9tuQHkE_Slipping Away.mp3"
    audio = load_wav(track_path, TARGET_SR, AUDIO_LEN)

    # extract audio embeddings from wav
    emb, ts = torchopenl3.get_audio_embedding(
        audio,
        TARGET_SR,
        model=embed_model,
        hop_size=HOP_SIZE,
        embedding_size=MUSIC_EMB_SIZE,
    )

    mean_emb = emb.mean(axis=1)

    # [1]
    # [1, 1, EMB]
    usr_idx = torch.tensor([34], dtype=torch.int32).to(DEVICE)
    batched_emb = mean_emb.unsqueeze(0)

    # [B, EMB]
    # [B, N, EMB]
    urs_x, embs, _ = align_model(usr_idx, batched_emb)

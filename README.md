# EMBEDDINGS

To generate the embeddings use embedgen.py

just define the path to the folder containing the audio, the stats csv file and the path to the folder where you want to save the embeddings

If needed redefine the `__getitem__` method in the MusicDataset class to fit the needs of your model

# FROM MP3 TO COSSIM

~~These steps are implemented in the `aligner.py` file, follow them to get from the mp3 file to the cosine similarity between the user embeddings and the music embeddings. User embeddings are stored in the model and correspond to the 1000 users in the Last.fm dataset (actually is a bit less than 1000 because not all users have listened to music avaiable to us).~~

## WE HAVE NOW SWITCHED TO MERT

Look at `test.py` for a quick example on how to load the alignment model

<!---
Load model and config from checkpoint

```python
LOAD = "usrembeds/checkpoints/run_20241107_201542_best.pt"
model_state, config, _ = Aligner.load_model(LOAD)
```

Load the Aligner model with the settings stored in the config

```python
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
```

Load the music encoder, in this case we are using OpenL3, we might swtich to MERT later on

```python
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
```

Load the mp3 file from disk

```python
track_path = "/your/music/folder/trap.mp3"
audio = load_wav(track_path, TARGET_SR, AUDIO_LEN)
```

Extract the embeddings from the audio and average over all frames

```python

# extract audio embeddings from wav
emb, ts = torchopenl3.get_audio_embedding(
    audio,
    TARGET_SR,
    model=embed_model,
    hop_size=HOP_SIZE,
    embedding_size=MUSIC_EMB_SIZE,
)

mean_emb = emb.mean(axis=1)
```

Reshape the user indexes tensor and the music embeddings tensor to the right shapes

```python
# [1]
# [1, 1, EMB]
usr_idx = torch.tensor([34], dtype=torch.int32).to(DEVICE)
batched_emb = mean_emb.unsqueeze(0)
```

Run the model with the user indexes and the music embeddings, you will
get the user embeddings and the projected music embeddings

```python
# [B, EMB]
# [B, N, EMB]
urs_x, embs, _ = align_model(usr_idx, batched_emb)
```

`N` is the number of songs for every batch, in this case we are only running one batch with one song
--->

FOR REPRODUCIBILITY IN CUDA: 
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
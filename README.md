# EMBEDDINGS

To generate the embeddings use embedgen.py

just define the path to the folder containing the audio, the stats csv file and the path to the folder where you want to save the embeddings

If needed redefine the `__getitem__` method in the MusicDataset class to fit the needs of your model

# ALIGNER

To load the Aligner model follow the code in test.py

Load model and config from a checkpoint

```python
LOAD = "usrembeds/checkpoints/run_20241107_201542_best.pt"
model_state, config, _ = Aligner.load_model(LOAD)
```

load the dataset and dataloader

```python

# you rly only need these two in training
NEG = config["neg_samples"] # number of negative samples
MUL = config["multiplier"]  # multiplier for the dataset

membs_path = "usrembeds/data/embeddings/batched"
save_path = "usrembeds/checkpoints"

dataset = ContrDataset(
    membs_path,
    stats_path,
    nneg=NEG,
    multiplier=MUL,
    transform=None,
)

_, val_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

NUSERS = dataset.nusers

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=16, shuffle=True
)
```

Instantiate the model with the config and load the state

```python
EMB_SIZE = config["emb_size"]       # size of the user embeddings
MUSIC_EMB_SIZE = config["prj_size"] # size of the music embeddings (512 with OpenL3)
PRJ = config["prj"]                 # projection type

# these are only for training
TEMP = config["temp"]
LT = config["learnable_temp"]

model = Aligner(
    n_users=NUSERS,
    emb_size=EMB_SIZE,
    prj_size=MUSIC_EMB_SIZE,
    prj_type=PRJ,
    lt=LT,
    temp=TEMP,
).to(DEVICE)

model.load_state_dict(model_state)
```

Have fun with the model

```python
for tracks in tqdm(val_dataloader):

        # [B]               indexes of the users
        # [B, 1, EMB]       positive embeddings
        # [B, NNEG, EMB]    sets of negative embeddings
        # [B]               weights for the loss
        idx, posemb, negemb, weights = tracks

        idx = idx.to(DEVICE)
        posemb = posemb.to(DEVICE)
        negemb = negemb.to(DEVICE)
        weights = weights.to(DEVICE)

        # if you don't care about positive/negative samples and you just need
        # the user embeddings and the projected music embeddings you just give
        # the model the user indexes [B] and your sets of music embeddings [B, N, EMB]
        allemb = torch.cat((posemb, negemb), dim=1)

        urs_x, embs, _ = model(idx, allemb)

        # urs_x is the user embeddings [B, EMB]
        # embs is the projected music embs [B, N, EMB]
```

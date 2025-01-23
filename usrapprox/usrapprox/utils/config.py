from dataclasses import dataclass

@dataclass
class AlignerV2Config:
    # Used only to correctly load alignerv2 model
    emb_size: int = 256
    batch_size: int = 64
    neg_samples: int = 20
    temp: float = 0.2
    learnable_temp: bool = True
    multiplier: int = 10
    weight: int = 0
    prj: str = "shared"
    aggr: str = "gating-tanh"
    nusers: int = 967
    prj_size: int = 768
    hidden_size: int = 2048
    drop: float = 0.2
    lr: float = 0.001
    noise_level: float = 0.0
    encoder: str = "MERT"
    abs_file_path: str = "usrembeds/checkpoints/run_20241227_151619_best.pt" 
    
@dataclass
class UserConfig:
    memory_length: int
    amount:int # number of users
    init: str = "mean" # "random" or "mean" or "rmean"
    rmean: float = 0.1 # used only if init is "rmean" - weight of the random noise

@dataclass
class TrainConfig:
    # USED ONLY FOR TEST TRAINING (training with offline data)
    splits_path: str = "usrembeds/data/splits.json"
    embs_path: str = "usrembeds/data/embeddings/embeddings_full_split"
    stats_path: str = "usrembeds/data/clean_stats.csv"  # used only by ContrDatasetMERT
    npos: int = 4
    nneg: int = 4
    batch_size: int = 128
    num_workers: int = 10
    multiplier: int = 50  # used only by ContrDatasetMert
    type: str = "asd"  # ContrDatasetMERT or anything
    epochs: int = 20

    # COMMON
    lr: float = 0.001

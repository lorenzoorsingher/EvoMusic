import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass, field

import yaml


def dataclass(*args, **kwargs):
    """
    Creates a dataclass that can handle nested dataclasses
    and automatically convert dictionaries to dataclasses.
    """

    def wrapper(cls):
        cls = og_dataclass(cls, **kwargs)
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            for name, value in kwargs.items():
                field_type = cls.__annotations__.get(name, None)
                if is_dataclass(field_type) and isinstance(value, dict):
                    new_obj = field_type(**value)
                    kwargs[name] = new_obj
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls

    return wrapper(args[0]) if args else wrapper


@dataclass
class LLavaConfig:
    model: str


@dataclass
class ReasonSegDatasetConfig:
    json_path: str
    image_dir: str
    mask_dir: str

    def __post_init__(self):
        self.json_path = os.path.expanduser(self.json_path)
        self.image_dir = os.path.expanduser(self.image_dir)
        self.mask_dir = os.path.expanduser(self.mask_dir)

@dataclass
class COCODatasetConfig:
    dataset_zoo_dir: str
    num_samples: int = 1000
    name: str = "coco-2017"

    def __post_init__(self):
        self.dataset_zoo_dir = os.path.expanduser(self.dataset_zoo_dir)

@dataclass
class DatasetConfig:
    ReasonSeg: ReasonSegDatasetConfig
    COCO: COCODatasetConfig

@dataclass
class SAMConfig:
    model: str
    checkpoint_dir: str
    resize: int
    n_masks: int

    def __post_init__(self):
        self.checkpoint_dir = os.path.expanduser(self.checkpoint_dir)


@dataclass
class AlphaCLIPConfig:
    model: str
    checkpoint_dir: str

    def __post_init__(self):
        self.checkpoint_dir = os.path.expanduser(self.checkpoint_dir)

@dataclass
class OthersConfig:
    wandb_token: str

@dataclass
class OptimizerConfig:
    adapter_lr: float
    lora_lr: float

@dataclass
class schedulerConfig:
    eta_min: float

@dataclass
class AdapterParams:
    expand_factor: int = 2
    noise_level: float = 0
    mlp_adapter: bool = True
    num_linears: int = 25
    hidden_dim: int = None
    num_heads: list[int] = field(default_factory=lambda: [1])
    num_queries: list[int] = field(default_factory=lambda: [10])

@dataclass
class ModelParams:
    adapter_params: AdapterParams
    lora_rank: int = 16
    q4: bool = True
    q8: bool = False
    mask_prob: float = 0
    dropout: float = 0.1
    pos_weight: float = 1
    neg_weight: float = 1
    temperature: float = 1
    end_turn_token: str = "<end_of_turn>\n"
    seg_pos: str = "before"
    text: bool = True
    
    
    
@dataclass
class TrainSplitDataConfig:
    COCO: str = None
    ReasonSeg: str = None
    
    def __post_init__(self):
        self.COCO = os.path.expanduser(self.COCO) if self.COCO else None
        self.ReasonSeg = os.path.expanduser(self.ReasonSeg) if self.ReasonSeg else None
    
@dataclass
class TrainingDataConfig:
    train_jsonl: TrainSplitDataConfig
    val_jsonl: str
    test_jsonl: str
    batch_size: int = 4
    
    def __post_init__(self):
        self.val_jsonl = os.path.expanduser(self.val_jsonl)
        self.test_jsonl = os.path.expanduser(self.test_jsonl)

@dataclass
class PreprocessConfig:
    only_mask: bool = False
    model: str = "alpha-clip"

@dataclass
class ExperimentConfig:
    epochs: int
    skip_test_val: bool
    optimizer: OptimizerConfig
    scheduler: schedulerConfig
    model_params: ModelParams
    preprocess: PreprocessConfig
    log_interval: int
    val_every: int
    dataset: TrainingDataConfig
    

@dataclass
class TrainingExperimentsConfig:
    experiments: dict[str, ExperimentConfig]
    
    def __post_init__(self):
        self.experiments = {k: ExperimentConfig(**v) for k, v in self.experiments.items()}
        
@dataclass
class ProjectConfig:
    llava: LLavaConfig
    dataset: DatasetConfig
    sam: SAMConfig
    alphaclip: AlphaCLIPConfig
    train: TrainingExperimentsConfig
    others: OthersConfig
    # just need to add stuff here and define what is here



def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))

if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)
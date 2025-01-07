import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass, field
import torch

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
class MusicGeneratorConfig:
    model: str
    input_type: str = "text"
    output_dir: str = "output"
    exp_name: str = "default"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert self.input_type in [
            "text",
            "token_embeddings",
            "embeddings",
        ], "input_type must be either 'text', 'token_embedding' or 'embeddings'"


@dataclass
class EasyRiffusionConfig(MusicGeneratorConfig):
    model: str = "riffusion/riffusion-model-v1"
    inference_steps: int = 50


@dataclass
class ProjectConfig:
    music_generator: MusicGeneratorConfig
    riffusion_pipeline: EasyRiffusionConfig
    # just need to add stuff here and define what is here


def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)

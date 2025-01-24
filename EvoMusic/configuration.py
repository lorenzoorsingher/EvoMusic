import os
from dataclasses import dataclass as og_dataclass
from dataclasses import is_dataclass, field
import torch

import wandb.util
import yaml
import wandb


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


# =================================================================================================
# Music Generation configuration
# =================================================================================================

@dataclass
class MusicGeneratorConfig:
    model: str
    input_type: str = "text"
    output_dir: str = "output"
    name: str = "default"
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


# =================================================================================================
# Music Evolution configuration
# =================================================================================================


@dataclass
class evolutionLogger:
    wandb: bool = False  # whether to log the evolution to the wandb platform
    project: str = ""  # name of the wandb project
    name: str = ""  # name of the wandb run
    # NOTE: the name is overriden by the experiment name

    wandb_token: str = ""  # wandb token to use for the logging

    visualizations: bool = False  # whether to visualize the evolution

    def __post_init__(self):
        if self.wandb:
            wandb.login(key=self.wandb_token)


@dataclass
class LLMConfig:
    api_key: str  # openai api key
    temperature: float = 0.7
    model: str = "gpt-4o-mini"
    api_uri: str = "https://api.openai.com/v1/chat/completions"


@dataclass
class FitnessConfig:
    mode: str = "user"  # can either be user, music or dynamic
    target_user: int = 0  # static target user for mode user
    target_music: str = ""  # path to the target song
    noise_weight: float = 0.1  # weight of the noise in the fitness

    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # device to use for the fitness evaluation

    def __post_init__(self):
        assert self.mode in ["user", "music", "dynamic"], "Invalid fitness mode"
        if self.target_music != "":
            self.target_music = os.path.expanduser(self.target_music)


@dataclass
class LLMPromptOperator:
    name: str
    input: int  # number of individuals to input
    output: int  # number of individuals to output
    prompt: str
    probability: float  # NOTE: when the operator is not applied a random sample is used

    def __post_init__(self):
        assert self.input > 0, "Input size must be greater than 0"
        assert self.output > 0, "Output size must be greater than 0"
        assert (
            self.probability >= 0 and self.probability <= 1
        ), "Probability must be in the range [0,1]"

        # if output is greater than input then probability must be 1
        assert (
            self.output <= self.input or self.probability == 1
        ), "Probability must be 1 when output is greater than input"


@dataclass
class searchConf:
    mode: str
    """
        modes are the following:
        - "full LLM" : full LLM search, the current population with their fitness is passed to the LLM and we directly task him to generate the next population after reasoning
        - "LLM evolve" : the LLM is used to generate the next population by using base generic operators (crossover, mutation)
        - "CMAES", "PGPE", "XNES", "SNES", "CEM" : use the algorithms in the evotorch library
        - "CoSyNE", "GA" : use the algorithms in the evotorch library

        NOTE:   the LLM evolve mode is only available when using the prompt optimization
                while the last modes are available when using the embeddings optimization
    """

    # general search parameters
    population_size: int = 10  # size of the population

    sample: bool = (
        False  # use multinomial sampling to select the individuals for all operations
    )
    temperature: float = 1  # temperature for the sampling
    novel_prompts: float = 0  # fraction of poupulation to create ex-novo, range [0,1]
    elites: float = 0  # fraction of population to keep from the previous, range [0,1]

    # full LLM parameters
    full_LLM_prompt: str = ""
    """
        Prompt to use when using the full LLM mode
        You have access to the following special tokens:
        - {ranking} : the current population rankin with their score
        - {num_generate}: the number of individuals to generate
    """

    # LLM evolve parameters
    LLM_genetic_operators: list[LLMPromptOperator] = field(default_factory=list)
    """
        Genetic operators to use when using the LLM evolve mode.
        You can create operators that apply multiple opearations at the same time by describing what you want the LLM to do.
        Do not use anywhere the <prompt> </prompt> tags, as they are used to extract the final output from the LLM.
        You can define where to add the prompts by using {prompts} token.
        NOTE: the result from the previous operator is passed to the next one, thus they need to have a compatible output and input size
    """
    tournament_size: int = 2  # size of the tournament for selection

    # evotorch parameters
    evotorch: dict = field(
        default_factory=dict
    )  # additional parameters for the search algorithm when using evotorch's algorithms

    def __post_init__(self):
        avail_modes = [
            "full LLM",
            "LLM evolve",
            "CMAES",
            "PGPE",
            "XNES",
            "SNES",
            "CEM",
            "CoSyNE",
            "GA",
        ]
        assert self.mode in avail_modes, "Invalid search mode"

        if self.mode == "LLM evolve":
            # convert list of dict to list of LLMPromptOperator
            self.LLM_genetic_operators = [
                LLMPromptOperator(**operator) for operator in self.LLM_genetic_operators
            ]

            assert self.tournament_size >= self.LLM_genetic_operators[0].input

            # check operators compatibility
            output = None
            for operator in self.LLM_genetic_operators:
                if output is not None:
                    assert (
                        operator.input == output
                    ), f"Operator { operator.name } input size is not compatible with the previous operator output size"
                output = operator.output

            assert (
                self.population_size % output == 0
            ), "Population size must be a multiple of the output size of the last operator"


@dataclass
class evoConf:
    exp_name: str  # name of the experiment
    generations: int  # number of generations to run the evolution

    search: searchConf
    fitness: FitnessConfig
    LLM: LLMConfig
    logger: evolutionLogger

    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # device to use for the evolution
    max_seq_len: int = (
        5  # maximum length of the prompt when using the token or embeddings optimization
    )
    duration: int = 5  # duration of the generated music in seconds
    best_duration: int = 10  # duration of the best solution in seconds

    evotorch: dict = field(
        default_factory=dict
    )  # additional parameters for the evolution algorithm when using evotorch's algorithms

    def __post_init__(self):
        if self.output_dir != "":
            self.output_dir = os.path.expanduser(self.output_dir)

        self.logger.name = self.exp_name + "_" + wandb.util.generate_id()


# =================================================================================================
# User Approximation configuration
# =================================================================================================

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
    amount: int  # number of users
    init: str = "mean"  # "random" or "mean" or "rmean"
    rmean: float = 0.1  # used only if init is "rmean" - weight of the random noise


@dataclass
class TrainConfig:
    # USED ONLY FOR TEST TRAINING (training with offline data)
    splits_path: str = "usrembeds/data/splits.json"
    embs_path: str = "usrembeds/data/embeddings/embeddings_full_split"
    stats_path: str = "usrembeds/data/clean_stats.csv"  # used only by ContrDatasetMERT
    npos: int = 1
    nneg: int = 4
    batch_size: int = 128
    num_workers: int = 10
    multiplier: int = 50  # used only by ContrDatasetMert
    type: str = "ContrDatasetMERT"  # ContrDatasetMERT or anything
    epochs: int = 20

    # COMMON
    lr: float = 0.001

@dataclass
class UserApproximationConfig:
    aligner: AlignerV2Config
    user: UserConfig

    def __post_init__(self):
        assert self.user.init in ["random", "mean", "rmean"], "Invalid init mode"


# =================================================================================================
# Project configuration
# =================================================================================================

@dataclass
class ProjectConfig:
    evolution: evoConf = None  # evolution configuration

    music_model: str = "musicgen"  # can either be "musicgen" or "riffusion"
    music_generator: MusicGeneratorConfig = None  # use this for musicgen model
    riffusion_pipeline: EasyRiffusionConfig = (
        None  # define this when using riffusion model
    )

    def __post_init__(self):
        assert self.music_model in [
            "musicgen",
            "riffusion",
        ], "music_model must be either 'musicgen' or 'riffusion'"
        assert (
            self.music_model == "musicgen" and self.music_generator is not None
        ) or (
            self.music_model == "riffusion" and self.riffusion_pipeline is not None
        ), "music_generator or riffusion_pipeline must be defined"

        # check consistency of method used for generation and search
        if self.evolution:
            generation_conf: MusicGeneratorConfig = (
                self.music_generator
                if self.music_model == "musicgen"
                else self.riffusion_pipeline
            )
            generation_mode = generation_conf.input_type

            if generation_mode == "text":
                assert self.evolution.search.mode in [
                    "full LLM",
                    "LLM evolve",
                ], "search mode must be Full LLM or LLM evolve when using text input"
            else:
                assert self.evolution.search.mode not in [
                    "full LLM",
                    "LLM evolve",
                ], "search mode cannot be Full LLM or LLM evolve when using embeddings or token embeddings input"


def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)

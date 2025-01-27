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
    memory_length: int = 0  # number of tracks to remember
    minibatch: bool = False
    amount: int = 1 # number of users
    init: str = "mean"  # "random" or "mean" or "rmean"
    rmean: float = 0.1  # used only if init is "rmean" - weight of the random noise

    
    def __post_init__(self):
        assert self.init in ["random", "mean", "rmean"], "Invalid init mode"


@dataclass
class TrainConfig:
    splits_path: str = "usrembeds/data/splits.json"
    embs_path: str = "usrembeds/data/embeddings/embeddings_full_split"
    stats_path: str = "usrembeds/data/clean_stats.csv"  # used only by ContrDatasetMERT
    npos: int = 1
    nneg: int = 4
    batch_size: int = 128
    # num_workers: int = 10
    multiplier: int = 50  # used only by ContrDatasetMert
    type: str = "UserDefinedContrastiveDataset"  # UserDefinedContrastiveDataset (use this one) or ContrDatasetMERT
    epochs: int = 20
    random_pool: bool = None

    # COMMON
    lr: float = 0.001

@dataclass
class UserDefinition:
    user_type: str
    target_user_id: int = -1

    def __post_init__(self):
        assert self.user_type in ["real", "synth"], "Invalid user type"
        if self.user_type == "synth":
            assert self.target_user_id != -1, "Synth user must have a target user id"

@dataclass
class UserApproximationConfig:
    users: list[UserDefinition]
    aligner: AlignerV2Config = AlignerV2Config()
    user_conf: UserConfig = UserConfig()
    train_conf: TrainConfig = TrainConfig()
    
    best_solutions: int = 10 # number of best solutions to keep
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        assert len(self.users) == self.user_conf.amount, "Number of user types must match the number of users"
        assert self.best_solutions > 0, "Number of best solutions must be greater than 0"
        
        self.users = [UserDefinition(**user) for user in self.users]


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
    target_music: str = ""  # path to the target song
    noise_weight: float = 0.25  # weight of the noise in the fitness
    
    device: str = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # device to use for the fitness evaluation

    def __post_init__(self):
        assert self.mode in ["user", "music", "dynamic"], "Invalid fitness mode"
        if self.target_music != "":
            self.target_music = os.path.expanduser(self.target_music)
        
        if self.mode == "music":
            assert os.path.exists(self.target_music), "Target music file does not exist"

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
class GAoperator:
    name: str
    """
        name of the operator, can be one of the following:
        - "CosynePermutation" : apply a permutation to the individuals, requires a tournament_size parameter
        - "OnePointCrossOver" : apply a one point crossover to the individuals, requires a tournament_size parameter
        - "MultiPointCrossOver" : apply a multi point crossover to the individuals, requires a tournament_size parameter
        - "GaussianMutation" : apply a gaussian mutation to the individuals, requires a stdev parameter
        - "PolynomialMutation" : apply a polynomial mutation to the individuals, requires a eta parameter 
        - "SimulatedBinaryCrossOver" : apply a simulated binary crossover to the individuals, requires a eta parameter
        - "TwoPointCrossOver" : apply a two point crossover to the individuals, requires a tournament_size parameter
    """
    parameters: dict
    
    def __post_init__(self):
        assert self.name in ["CosynePermutation", "OnePointCrossOver", "MultiPointCrossOver", "GaussianMutation", "PolynomialMutation", "SimulatedBinaryCrossOver", "TwoPointCrossOver"], "Invalid operator name"
        if self.name in ["OnePointCrossOver", "MultiPointCrossOver", "SimulatedBinaryCrossOver", "TwoPointCrossOver"]:
            assert "tournament_size" in self.parameters, f"tournament_size must be defined for this operator {self.name}"
            
        if self.name == "GaussianMutation":
            assert "stdev" in self.parameters, f"stdev Standard deviation must be defined for this operator {self.name}"
            
        if self.name == "SimulatedBinaryCrossOver":
            assert "eta" in self.parameters, f"eta must be defined for this operator {self.name}"
    
@dataclass
class searchConf:
    mode: str
    """
        modes are the following:
        - "full LLM" : full LLM search, the current population with their fitness is passed to the LLM and we directly task him to generate the next population after reasoning
        - "LLM evolve" : the LLM is used to generate the next population by using base generic operators (crossover, mutation)
        - "CMAES", "PGPE", "XNES", "SNES", "CEM" : use the algorithms in the evotorch library
        - "GA" : use the algorithms in the evotorch library, uses the GA operators defined in GA_operators + the parameters in evotorch for the search

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

    # GA parameters
    GA_operators: list[GAoperator] = field(default_factory=list)

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

            if (self.population_size-self.novel_prompts*self.population_size-self.population_size*self.elites) % output != 0:
                print(f"Warning: population size is not divisible by the output size of the last operator")
                
        elif self.mode == "GA":
            assert len(self.GA_operators) > 0, "At least one operator must be defined"
            # convert list of dict to list of GAoperator
            self.GA_operators = [
                GAoperator(**operator) for operator in self.GA_operators
            ]


@dataclass
class evoConf:
    exp_name: str  # name of the experiment
    generations: int  # number of generations to run the evolution

    search: searchConf
    fitness: FitnessConfig
    logger: evolutionLogger
    
    initialization: str = "LLM"  # initialization mode, can be "LLM" or "file"
    """
        Initialization mode:
        - "LLM" : initialize the population using the LLM (need to specify the LLM)
        - "file" : initialize the population using a file containing the prompts separated by a new line (need to specify the init_file)
    """
    init_file: str = "EvoMusic/music_generation/init_prompts.txt"  # path to the initialization file when using the "file" initialization
    LLM: LLMConfig = None  # LLM configuration when using the LLM initialization

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
        self.logger.name = self.exp_name + "_" + wandb.util.generate_id()
        
        if self.initialization == "file":
            assert os.path.exists(self.init_file), "Initialization file does not exist"
        elif self.initialization == "LLM":
            assert self.LLM is not None, "LLM configuration must be defined when using LLM initialization"
            
        assert self.generations > 0, "Number of generations must be greater than 0"
        assert self.max_seq_len > 0, "Maximum sequence length must be greater than 0"
        assert self.duration > 0, "Duration must be greater than 0"
        assert self.best_duration > 0, "Best duration must be greater than 0"
        
        if self.search.mode in ["LLM evolve", "full LLM"]:
            assert self.LLM is not None, "LLM configuration must be defined when using LLM evolve or full LLM search mode"


# =================================================================================================
# Project configuration
# =================================================================================================

@dataclass
class ProjectConfig:
    epochs: int = 1  # maximum number of epochs to run the pipeline
    evolution: evoConf = None  # evolution configuration

    music_model: str = "musicgen"  # can either be "musicgen" or "riffusion"
    music_generator: MusicGeneratorConfig = None  # use this for musicgen model
    riffusion_pipeline: EasyRiffusionConfig = (
        None  # define this when using riffusion model
    )
    
    user_model: UserApproximationConfig = None  # user model configuration

    def __post_init__(self):
        assert self.epochs > 0, "Number of epochs must be greater than 0"
        
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
                
            if self.evolution.fitness.mode in ["user", "dynamic"]:
                assert self.user_model is not None, "user_model must be defined when using user or dynamic fitness mode"
                
            assert self.evolution.search.population_size >= self.user_model.best_solutions, "Population size must be greater than the number of best solutions to generate for finetuning"

def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)

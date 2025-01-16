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


@dataclass
class evolutionLogger:
    wandb: bool = False # whether to log the evolution to the wandb platform
    project: str = "" # name of the wandb project
    name: str = "" # name of the wandb run
    # NOTE: the name is overriden by the experiment name
    
    wandb_token: str = "" # wandb token to use for the logging
    
    visualizations: bool = False # whether to visualize the evolution
    
    def __post_init__(self):
        if self.wandb:
            wandb.login(key=self.wandb_token)
    

@dataclass
class LLMConfig:
    api_key: str # openai api key
    temperature: float = 0.7
    model: str = "gpt-4o-mini"
    api_uri : str = "https://api.openai.com/v1/chat/completions"

@dataclass
class FitnessConfig:
    mode: str = "user" # can either be user, music or dynamic
    target_user: int = 0 # static target user for mode user
    target_music: str = "" # path to the target song 
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # device to use for the fitness evaluation
    
    def __post_init__(self):
        assert self.mode in ["user", "music", "dynamic"], "Invalid fitness mode"
        if self.target_music != "":
             self.target_music = os.path.expanduser(self.target_music)

@dataclass
class LLMPromptOperator:
    name: str
    input: int # number of individuals to input
    output: int # number of individuals to output
    description: str
    probability: float
    """
        genetic operator to add to the LLM prompt evolution
    """

@dataclass
class searchConf:
    mode: str
    """
        modes are the following:
        - "full LLM" : full LLM search, the current population with their fitness is passed to the LLM and we directly task him to generate the next population after reasoning
        - "LLM evolve" : the LLM is used to generate the next population by using base generic operators (crossover, mutation)
        - "CMAES", "PGPE", "XNES", "SNES", "CEM" : use the algorithms in the evotorch library

        NOTE:   the LLM evolve mode is only available when using the prompt optimization
                while the last modes are available when using the embeddings optimization
    """
    
    # general search parameters
    population_size: int = 10 # size of the population
    
    sample: bool = False # use multinomial sampling to select the individuals for all operations
    novel_prompts: float = 0 # fraction of poupulation to create ex-novo, range [0,1]
    elites : float = 0 # fraction of population to keep from the previous, range [0,1]
    
    # full LLM parameters
    
    # LLM evolve parameters
    LLM_genetic_operators: list[LLMPromptOperator] = field(default_factory=list) # genetic operators to use when using the LLM evolve mode
    tournament_size: int = 2 # size of the tournament for selection
    
    # evotorch parameters
    evotorch: dict = field(default_factory=dict) # additional parameters for the search algorithm when using evotorch's algorithms
    
    def __post_init__(self):
        assert self.mode in ["full LLM", "LLM evolve", "CMAES", "PGPE", "XNES", "SNES", "CEM"], "Invalid search mode"

@dataclass
class evoConf:
    exp_name: str # name of the experiment
    generations: int # number of generations to run the evolution
    output_dir: str # path to directory where the generated music will be saved
    
    
    search: searchConf
    fitness: FitnessConfig
    LLM: LLMConfig
    logger: evolutionLogger
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # device to use for the evolution
    max_seq_len: int = 5 # maximum length of the prompt when using the token or embeddings optimization
    duration: int = 5 # duration of the generated music in seconds
    
    evotorch: dict = field(default_factory=dict) # additional parameters for the evolution algorithm when using evotorch's algorithms
    
    def __post_init__(self):
        if self.output_dir != "":
            self.output_dir = os.path.expanduser(self.output_dir)
            
        self.logger.name = self.exp_name + "_" + wandb.util.generate_id()

@dataclass
class ProjectConfig:
    evolution : evoConf = None# evolution configuration
    
    music_model : str = "musicgen" # can either be "musicgen" or "riffusion"
    music_generator: MusicGeneratorConfig = None # use this for musicgen model
    riffusion_pipeline: EasyRiffusionConfig = None # define this when using riffusion model

    def __post_init__(self):
        assert self.music_model in ["musicgen", "riffusion"], "music_model must be either 'musicgen' or 'riffusion'"
        assert (self.music_model == "musicgen" and self.music_generator is not None) or (self.music_model == "riffusion" and self.riffusion_pipeline is not None), "music_generator or riffusion_pipeline must be defined"
        
        # check consistency of method used for generation and search
        if self.evolution:
            generation_conf: MusicGeneratorConfig = self.music_generator if self.music_model == "musicgen" else self.riffusion_pipeline
            generation_mode = generation_conf.input_type
            
            if generation_mode == "text":
                assert self.evolution.search.mode in ["full LLM", "LLM evolve"], "search mode must be Full LLM or LLM evolve when using text input"
            else:
                assert self.evolution.search.mode not in ["full LLM", "LLM evolve"], "search mode cannot be Full LLM or LLM evolve when using embeddings or token embeddings input"
            

def load_yaml_config(path) -> ProjectConfig:
    with open(path) as file:
        return ProjectConfig(**yaml.load(file, Loader=yaml.FullLoader))


if __name__ == "__main__":
    config = load_yaml_config("config.yaml")
    print(config)

from diffusers.utils.testing_utils import enable_full_determinism
from EvoMusic.evolution.evolve import MusicEvolver
from EvoMusic.configuration import load_yaml_config
from music_generation.generators import EasyRiffPipeline, MusicGenPipeline

enable_full_determinism()

import torch
import numpy as np
import random

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.seed_all()
    
    
# Load environment variables
config = load_yaml_config("config.yaml")

# ------------------------- Music Generation Setup ------------------------
if config.music_model == "riffusion":
    music_generator = EasyRiffPipeline(config.riffusion_pipeline)
elif config.music_model == "musicgen":
    music_generator = MusicGenPipeline(config.music_generator)
else:
    raise ValueError(
        "Invalid music model specified. Choose between 'musicgen' and 'riffusion'."
    )

# Evolve prompts or embeddings
evolver = MusicEvolver(config.evolution, music_generator)
results = evolver.evolve(n_generations=config.evolution.generations)

# Save the best solution
best_sol = results["best_solution"]["solution"]
best_fitness = results["best_solution"]["fitness"]
    
best_audio_path = music_generator.generate_music(
    input=best_sol, 
    name="BestSolution", 
    duration=config.evolution.duration
)
print(f"Best solution saved at: {best_audio_path}")
    
print(f"Best Fitness: {best_fitness}")

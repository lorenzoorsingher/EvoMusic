from evotorch.algorithms import CMAES, PGPE, XNES, SNES, CEM

from music_generation.generators import MusicGenerator
from EvoMusic.evolution.searchers import PromptSearcher, MusicOptimizationProblem
from EvoMusic.evolution.logger import LivePlotter
from EvoMusic.configuration import evoConf

class MusicEvolver:
    def __init__(self, config: evoConf, music_generator: MusicGenerator):
        self.config = config
        self.music_generator = music_generator

        self.problem = MusicOptimizationProblem(config, music_generator)

        if self.problem.text_mode:
            # Initialize the custom PromptSearcher optimizer
            self.optimizer = PromptSearcher(self.problem, config.search, config.LLM)
        else:
            # Initialize the optimizer for embedding optimization
            if config.search.mode == "CMAES":
                self.optimizer = CMAES(
                    self.problem, stdev_init=1, popsize=config.search.population_size
                )
            elif config.search.mode == "PGPE":
                self.optimizer = PGPE(
                    self.problem,
                    popsize=config.search.population_size,
                    center_learning_rate=1,
                    stdev_learning_rate=1,
                    stdev_init=1,
                )
            elif config.search.mode == "XNES":
                self.optimizer = XNES(
                    self.problem, popsize=config.search.population_size, stdev_init=1
                )
            elif config.search.mode == "SNES":
                self.optimizer = SNES(
                    self.problem, popsize=config.search.population_size, stdev_init=1
                )
            elif config.search.mode == "CEM":
                self.optimizer = CEM(
                    self.problem, popsize=config.search.population_size, stdev_init=1
                )
            else:
                raise ValueError(
                    "Invalid searcher specified. Choose between 'CMAES', 'PGPE', 'XNES', 'SNES', 'CEM'."
                )

        # Run the evolution strategy
        print("Starting evolution...")
        LivePlotter(
            self.optimizer,
            self.problem,
            music_generator,
            {
                "search_conf": config.search.__dict__,
                "fitness_conf": config.fitness.__dict__,
                "generation_conf": music_generator.config.__dict__,
                "LLM": {
                    "model": config.LLM.model,
                    "temperature": config.LLM.temperature,
                },
                "evotorch": config.evotorch,
            },
            config.logger,
        )
        
    def evolve(self, n_generations: int=None):
        """
        Run the evolution strategy for a specified number of generations
        
        Args:
            n_generations (int): The number of generations to run the evolution for.
                If None, the number of generations specified in the configuration is used.
                
        Returns:
            dict: A dictionary containing the best solution and the last generation
                { "best_solution": { "fitness": float, "solution": list }, "last_generation": { "solutions": list, "fitness_values": list } }
        """
        if not n_generations:
            n_generations = self.config.generations
        self.optimizer.run(num_generations=n_generations)
        
        # Get the best solution
        best_fitness = self.optimizer.status["pop_best_eval"]
        best_sol = self.optimizer.status["pop_best"].values
        print("\n--- Evolution Complete ---")
        
        last_gen = self.optimizer.population.values
        last_gen = [last_gen[i] for i in range(len(last_gen))]
        last_gen_evals = self.optimizer.population.evals.view(-1).tolist()
        
        return {
            "best_solution": {
                "fitness": best_fitness,
                "solution": best_sol                  
            },
            "last_generation": {
                "solutions": last_gen,
                "fitness_values": last_gen_evals
            }
        }


if __name__ == "__main__":
    from diffusers.utils.testing_utils import enable_full_determinism
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

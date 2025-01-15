from evotorch.algorithms import CMAES, PGPE, XNES, SNES, CEM

if __name__ == "__main__":
    import sys
    sys.path.append("./")
    sys.path.append("../")

from music_generation.generators import MusicGenerator
from evolution.searchers import PromptSearcher, MusicOptimizationProblem
from evolution.logger import LivePlotter
from configuration import evoConf

def evolve_prompts(config:evoConf, music_generator: MusicGenerator):
    """
    Evolve prompts or embeddings to maximize similarity to target embedding using Evotorch's CMA-ES or custom Searcher.
    """
    problem = MusicOptimizationProblem(config, music_generator)
    
    if problem.text_mode:
        # Initialize the custom PromptSearcher optimizer
        optimizer = PromptSearcher(problem, config.search, config.LLM)
    else:
        # Initialize the optimizer for embedding optimization
        if config.search.mode == "CMAES":
            optimizer = CMAES(problem, stdev_init=1, popsize=config.search.population_size)
        elif config.search.mode == "PGPE":
            optimizer = PGPE(
                problem,
                popsize=config.search.population_size,
                center_learning_rate=1,
                stdev_learning_rate=1,
                stdev_init=1,
            )
        elif config.search.mode == "XNES":
            optimizer = XNES(problem, popsize=config.search.population_size, stdev_init=1)
        elif config.search.mode == "SNES":
            optimizer = SNES(problem, popsize=config.search.population_size, stdev_init=1)
        elif config.search.mode == "CEM":
            optimizer = CEM(problem, popsize=config.search.population_size, stdev_init=1)
        else:
            raise ValueError(
                "Invalid searcher specified. Choose between 'CMAES', 'PGPE', 'XNES', 'SNES', 'CEM'."
            )

    # Run the evolution strategy
    print("Starting evolution...")
    LivePlotter(optimizer, problem, "pop_best_eval")
    optimizer.run(num_generations=config.generations)

    # Get the best solution
    best_fitness = optimizer.status["pop_best_eval"]
    print("\n--- Evolution Complete ---")
    print(f"Best Fitness (Cosine Similarity): {best_fitness}")

    if problem.prompt_optim:
        # Decode the best solution into a prompt
        best_prompt = problem.prompts[optimizer._population.evals.argmax()]
        print(f"Best Prompt: '{best_prompt}'")
        return best_prompt, best_fitness
    else:
        # For embeddings, return the best embedding
        best_embedding = optimizer.status["pop_best"].values
        print(f"Best Embedding: {best_embedding}")
        return best_embedding, best_fitness


if __name__ == "__main__":    
    from configuration import load_yaml_config
    from music_generation.generators import EasyRiffPipeline, MusicGenPipeline
    # Load environment variables
    config = load_yaml_config("config.yaml")


    # ------------------------- Music Generation Setup ------------------------
    if config.music_model == "riffusion":
        music_generator = EasyRiffPipeline(config.riffusion_pipeline)
    elif config.music_model == "musicgen":
        music_generator = MusicGenPipeline(config.music_generator)
    else:
        raise ValueError("Invalid music model specified. Choose between 'musicgen' and 'riffusion'.")

    # Evolve prompts or embeddings
    best_solution, best_fitness = evolve_prompts(config.evolution, music_generator)

    # Generate final music using the best prompt or embedding
    print("\nGenerating final music with the best solution...")
    final_audio_path = music_generator.generate_music(inputs=best_solution, name="final_music", duration=config.evolution.duration)
    print(f"Final audio saved at: {final_audio_path}")

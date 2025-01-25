from evotorch.core import Problem, Solution
from EvoMusic.configuration import evoConf
from EvoMusic.music_generation.generators import MusicGenerator
from EvoMusic.evolution.fitness import MusicScorer
from EvoMusic.evolution.searchers import LLMPromptGenerator
import torch
import os
import time

class MusicOptimizationProblem(Problem):
    """
    Evotorch Problem for optimizing music prompts or embeddings.
    """
    evo_config: evoConf
    music_generator: MusicGenerator

    def __init__(
        self,
        evolutions_config: evoConf,
        music_generator: MusicGenerator,
    ):
        self.evo_config = evolutions_config
        self.text_mode = self.evo_config.search.mode in ["full LLM", "LLM evolve"]
        
        super().__init__(
            objective_sense="max",
            device=self.evo_config.device,
            solution_length=
                None if self.text_mode
                else self.evo_config.max_seq_len * music_generator.get_embedding_size(),
            dtype=object if self.text_mode else torch.float32,
        )
        
        self.evaluator = MusicScorer(self.evo_config.fitness)
        self.music_generator = music_generator
        self.LLM_model = LLMPromptGenerator(self.evo_config.LLM)
        
        self.sample_time = 0 # time taken to generate one sample in the population
        self.current_time = 0 # time taken to generate the current population
        
        self.generated = 0
        self.total_generated = 0
        
        self.epoch_pop = self.evo_config.search.population_size

    def _evaluate(self, solution: Solution):
        """
        Objective function that maps solution vectors to prompts or embeddings, generates music,
        computes embeddings, and evaluates similarity to the target embedding.
        """
        start_time = time.time()
        self.generated += 1

        generator_input = solution.values
        if not self.text_mode:
            # copy the input to a new tensor as the values are read-only
            generator_input = solution.values.clone().detach()
        audio_path = self.music_generator.generate_music(input=generator_input, duration=self.evo_config.duration, name=f"music_intermediate")

        # Compute the embedding of the generated music
        fitness = self.evaluator.compute_fitness([audio_path]).squeeze()

        # Clean up generated audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            # print(f"Deleted temporary audio file: {audio_path}")

        generation_time = time.time() - start_time
        if self.sample_time == 0: self.sample_time = generation_time
        else: self.sample_time = self.sample_time * 0.9 + generation_time * 0.1
        self.current_time += generation_time
        time_left = self.sample_time * (self.epoch_pop - self.generated)
        total_time = self.current_time + time_left
        # make into time format so it's easier to read
        total_time = time.strftime("%H:%M:%S", time.gmtime(total_time))
        current_time = time.strftime("%H:%M:%S", time.gmtime(self.current_time))
        
        if self.epoch_pop > 0:
            bar_length = 30
            filled_length = int(bar_length * self.generated // self.epoch_pop)
            bar = "█" * filled_length + "-" * (bar_length - filled_length)
            print(
                f"Generated {self.generated}/{self.epoch_pop} |{bar}| "
                f"{(100 * self.generated / self.epoch_pop):.1f}% "
                f"~ Fitness {fitness:.2f} "
                f"~ Progress {current_time} / {total_time} "
                f"~ Sample Time {generation_time:.2f}s",
                end="\r"
            )
        else:
            print(f"Generated {self.generated} | Fitness {fitness:.2f} | Sample Time {generation_time:.2f}s", end="\r")
            
        if self.generated >= self.epoch_pop and self.epoch_pop > 0:
            self.generated = 0
            self.total_generated += self.epoch_pop
            print(f"\nFinished generation for this population. Total Time: {self.current_time:.2f}s", end="\r")
            self.current_time = 0

        solution.set_evals(fitness)

    def _fill(self, values):
        prompts = []
        population = values.shape[0]

        print(
            f"Generating diverse prompts for the initial population of {population} solutions..."
        )

        processed_prompts = []

        while len(processed_prompts) < population:
            # Generate diverse prompts for the initial population
            prompts = self.LLM_model.generate_prompts(population)
            
            # if not in text mode, then check if the embeddings are valid
            if not self.text_mode:
                processed = self.music_generator.preprocess_text(prompts, self.evo_config.max_seq_len)
                processed = [prompt for prompt in processed if prompt.shape[0] == self.evo_config.max_seq_len]
            else:
                processed = prompts
            
            processed_prompts += processed[: population - len(processed_prompts)]
        
    
        if self.text_mode:
            # values is an object array
            for i,prompt in enumerate(prompts):
                values.set_item(i, prompt)
        else:
            processed_prompts = torch.stack(processed_prompts)
            values.copy_(processed_prompts.view(population, -1))
            
        return values

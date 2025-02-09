import random
from evotorch.core import Problem, Solution
from EvoMusic.configuration import evoConf, LLMConfig
from EvoMusic.music_generation.generators import MusicGenerator
from EvoMusic.evolution.fitness import MusicScorer
import torch
import os
import time
import requests
import json

class LLMPromptGenerator():
    def __init__(self, config: LLMConfig):
        self.config = config

    def query_llm(self, prompt: str):
        """
        Query the LLM API with the given prompt.
        """
        # print(f"\t[LLM] sent request to LLM")
        # print(f"Querying LLM with prompt: '{prompt}'")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        data = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You produce prompts used to generate music following the requests of the user. You should always respond with the requested prompts by encasing each one in <prompt> and </prompt> tags XML style. DO NOT USE THEM ANYWHERE ELSE. Examples of valid output prompts are:\n 1. <prompt> Text of prompt. </prompt>\n 2. <prompt> Another prompt. </prompt>",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
            "max_tokens": 5000,
        }
        try:
            response = requests.post(
                self.config.api_uri, headers=headers, data=json.dumps(data)
            )
            response.raise_for_status()
            llm_response = response.json()["choices"][0]["message"]["content"].strip()
            # print(f"LLM responded with: '{llm_response}'")
            # print(f"\t[LLM] API request successful")
            return llm_response
        except Exception as e:
            print(f"\t[LLM] API request failed: {e}")
            print(f"\t[LLM] waiting for 5 minutes before retrying...")
            time.sleep(300)
            return ""
        
    def parse_llm_response(self, response: str):
        """
        Parse the response from the LLM API and return the prompts.
        """
        prompts = []
        
        if "deepseek-r1" in self.config.model:
            #cut thinking part
            if "</think>" in response:
                response = response[response.index("</think>")+8:]
        
        if response.count("<prompt>") != response.count("</prompt>"):
            return prompts

        for answer in response.split("</prompt>"):
            if "<prompt>" in answer:
                prompts.append(answer[answer.index("<prompt>") + 8 :].strip())

        return prompts

    def generate_prompts(self, num_prompts: int):
        prompts = []
        while len(prompts) < num_prompts:
            answers = self.query_llm(
                f"Generate {num_prompts-len(prompts)} diverse prompts for generating music, they should span multiple generes, moods, ..."
            )

            if answers.count("<prompt>") != answers.count("</prompt>"):
                continue

            for answer in answers.split("</prompt>"):
                if "<prompt>" in answer:
                    prompts.append(answer[answer.index("<prompt>") + 8 :].strip())
                if len(prompts) >= num_prompts:
                    break

        return prompts

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

    def fill_with_LLM(self, population:int):
        """
            fill the population with diverse prompts generated by LLM
            
            Args:
                population: int, the size of the population
        """
        prompts = []
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
            print(f"Processed {len(processed_prompts)}/{population} prompts", end="\r")
        
        return processed_prompts
    
    def fill_with_file(self, population:int):
        """
            fill the population with prompts from a file
            
            Args:
                population: int, the size of the population
        """
        prompts_pool = []
        with open(self.evo_config.init_file, "r") as f:
            prompts_pool = f.readlines()
        # check if the pool is large enough
        if len(prompts_pool) < population:
            raise ValueError(f"Initial population size {population} is larger than the number of prompts in the init file {len(prompts_pool)}")
        
        processed_prompts = []
        while len(processed_prompts) < population:
            # randomly sample prompts from the pool
            prompts = random.sample(prompts_pool, population-len(processed_prompts))
            # strip "\n" from the prompts
            prompts = [prompt.strip() for prompt in prompts]
            
            # if not in text mode, then check if the embeddings are valid
            if not self.text_mode:
                processed = self.music_generator.preprocess_text(prompts, self.evo_config.max_seq_len)
                processed = [prompt for prompt in processed if prompt.shape[0] == self.evo_config.max_seq_len]
            else:
                processed = prompts
                
            processed_prompts += processed[: population - len(processed_prompts)]
            print(f"Processed {len(processed_prompts)}/{population} prompts", end="\r")
        
        return processed_prompts

    def _fill(self, values):
        population = values.shape[0]

        print(
            f"Generating diverse prompts for the initial population of {population} solutions..."
        )

        processed_prompts = []

        if self.evo_config.initialization == "LLM":
            processed_prompts = self.fill_with_LLM(population)
        else:
            processed_prompts = self.fill_with_file(population)
            
    
        if self.text_mode:
            # values is an object array
            for i,prompt in enumerate(processed_prompts):
                values.set_item(i, prompt)
        else:
            processed_prompts = torch.stack(processed_prompts)
            values.copy_(processed_prompts.view(population, -1))
            
        return values

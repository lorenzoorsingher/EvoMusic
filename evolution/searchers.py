import gc
import os
import requests
import json
import numpy as np
import random

from evotorch import Problem, Solution
from evotorch.algorithms import SearchAlgorithm
from evotorch.algorithms.searchalgorithm import SinglePopulationAlgorithmMixin
import torch

if __name__ == "__main__":
    import sys

    sys.path.append("./")
    sys.path.append("../")

from evolution.fitness import MusicScorer
from music_generation.generators import MusicGenerator
from configuration import searchConf, evoConf, LLMConfig

# ------------------------- Evolutionary Algorithm ------------------------
class LLMPromptGenerator():
    def __init__(self, config: LLMConfig):
        self.config = config

    def query_llm(self, prompt: str):
        """
        Query the LLM API with the given prompt.
        """
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
                    "content": "You produce prompts used to generate music following the requests of the user. You should always respond with only the requested prompts and by encasing each one of the **final** produced prompts in <prompt> and </prompt> tags. Like the followings:\n 1. <prompt> A music prompt. </prompt>\n 2. <prompt> Another music prompt. </prompt>",
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
            return llm_response
        except Exception as e:
            print(f"LLM API request failed: {e}")
            return "A default music prompt."
        
    def parse_llm_response(self, response: str):
        """
        Parse the response from the LLM API and return the prompts.
        """
        prompts = []
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

class PromptSearcher(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    def __init__(
        self,
        problem: Problem,
        search_config: searchConf,
        LLM_config: LLMConfig,
    ):
        SearchAlgorithm.__init__(self, problem)

        self._problem = problem

        self.config = search_config
        self.LLM_model = LLMPromptGenerator(LLM_config)

        self._population = None

        self.generations = 1

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self):
        return self._population

    @property
    def problem(self):
        return self._problem

    def get_elites(self) -> list[str]:
        """
        Get the elite solutions from the population.

        Args:
            indeces (list[int]): the indices of the elite solutions in the population sorted by fitness

        Returns:
            list[str]: the elite solutions
        """
        if self.config.elites == 0:
            return []

        indices = self.population.argsort()
        num_elites = int(self.config.elites * self.config.population_size)

        if self.config.sample:
            # sample elites based on their fitness
            fitness = self.population[indices].evals
            fitness = fitness - fitness.min() + 1e-6
            fitness = fitness / fitness.sum()
            fitness = fitness.view(-1).cpu().numpy()
            indices = np.random.choice(indices, num_elites, p=fitness, replace=False)
        else:
            indices = indices[:num_elites]
        
        return [self.population.values[i] for i in indices[:num_elites]]

    def get_novel_prompts(self) -> list[str]:
        """
        Get novel prompts for exploration.

        Returns:
            list[str]: the novel prompts
        """
        if self.config.novel_prompts == 0:
            return []

        num_novel = int(self.config.population_size * self.config.novel_prompts)
        novel_prompts = self.LLM_model.generate_prompts(num_novel)

        return novel_prompts

    def full_LLM_step(self):
        indices = self.population.argsort()
        best_idx = indices[0]
        pop_values = self.population.values
        pop_evals = [self.population[i].evals.item() for i in range(len(indices))]

        ranking = ""
        for i in indices:
            # limit to 2 decimal places
            ranking += f"{i+1}. {pop_values[i]} - {pop_evals[i]*50+50:.2f} / 100\n"

        best = pop_values[best_idx]
        print(f"Population Best: {best} - {pop_evals[best_idx]*50+50} / 100")

        # elites for exploitation
        new_prompts = self.get_elites()

        # novel prompts for exploration
        new_prompts += self.get_novel_prompts()

        # Update the population using LLM by giving it the scores of each prompt and asking for new ones
        while len(new_prompts) < self.config.population_size:
            LLM_prompt = self.config.full_LLM_prompt.format(
                ranking=ranking, 
                num_generate=self.config.population_size - len(new_prompts)
            )

            answers = self.LLM_model.query_llm(LLM_prompt)
            generated_prompts = self.LLM_model.parse_llm_response(answers)
            new_prompts += generated_prompts[: self.config.population_size - len(new_prompts)]
            
        # print("Current Population:\n\t- ", "\n\t- ".join(self._problem.prompts))
        # print("New Population:\n\t- ", "\n\t- ".join(new_prompts))
        print("Finished generating new prompts.")

        # Update the population
        self._population.set_values(new_prompts)

    def _step(self):
        """Perform a step of the solver"""
        # update the population
        if self._population is None:
            self._population = self._problem.generate_batch(self.config.population_size)
        elif self.config.mode == "full LLM":
            self.full_LLM_step()
        elif self.config.mode == "LLM evolve":
            raise NotImplementedError("LLM evolve mode is not yet implemented")
        else:
            raise ValueError("Invalid search mode")
        
        self._problem.evaluate(self.population)

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
            dtype=object if self.text_mode else torch.float16,
        )
        
        self.evaluator = MusicScorer(self.evo_config.fitness)
        self.music_generator = music_generator
        self.LLM_model = LLMPromptGenerator(self.evo_config.LLM)
        
        self.generated = 0

    def _evaluate(self, solution: Solution):
        """
        Objective function that maps solution vectors to prompts or embeddings, generates music,
        computes embeddings, and evaluates similarity to the target embedding.
        """
        self.generated += 1

        generator_input = solution.values
        if not self.text_mode:
            generator_input = generator_input.clone().detach()
        audio_path = self.music_generator.generate_music(input=generator_input, name=f"music_intermediate")

        # Compute the embedding of the generated music
        fitness = self.evaluator.compute_fitness([audio_path]).squeeze()

        # Clean up generated audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            # print(f"Deleted temporary audio file: {audio_path}")

        print(f"Generated: {self.generated} / {self.evo_config.search.population_size}")
        if self.generated >= self.evo_config.search.population_size:
            self.generated = 0
            print("Finished generation for this population.")

        solution.set_evals(fitness)

    def _fill(self, values):
        prompts = []
        population = values.shape[0]

        print(
            f"Generating diverse prompts for the initial population of {population} solutions..."
        )

        # Generate diverse prompts for the initial population
        prompts = self.LLM_model.generate_prompts(population)

        processed_prompts = self.music_generator.preprocess_text(prompts)
        
    
        if self.text_mode:
            # values is an object array
            for i,prompt in enumerate(prompts):
                values.set_item(i, prompt)
        else:
            values.copy_(processed_prompts.view(population, -1))
            
        return values

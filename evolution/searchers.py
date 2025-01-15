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

        # self._population = problem.generate_batch(self._population_size)

        self.best = None
        self.generations = 1

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self):
        return self._population

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

        indices = self._population.argsort()
        num_elites = int(self.config.elites * self.config.population_size)

        if self.config.sample:
            # sample elites based on their fitness
            fitness = self._population[indices].evals
            fitness = fitness - fitness.min() + 1e-6
            fitness = fitness / fitness.sum()
            indices = np.random.choice(indices, num_elites, p=fitness, replace=False)
            return [self._problem.prompts[indices[i]] for i in range(num_elites)]
        else:
            return [self._problem.prompts[indices[i]] for i in range(num_elites)]

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
        indices = self._population.argsort()

        ranking = ""
        for i in range(len(indices)):
            ranking += f"{i+1}. {self._problem.prompts[indices[i]]} - {self._population[indices[i]].evals.item()*50+50} / 100\n"

        best = self._problem.prompts[indices[0]]
        print(f"Population Best: {best} - {self._population[indices[0]].evals.item()*50+50} / 100")

        # elites for exploitation
        new_prompts = self.get_elites()

        # novel prompts for exploration
        new_prompts += self.get_novel_prompts()

        # Update the population using LLM by giving it the scores of each prompt and asking for new ones
        while len(new_prompts) < self._population_size:
            LLM_prompt = f"""Generate {self._population_size-len(new_prompts)} music prompts for generating music based on the classification and scores of the previous prompts. 
You should balance exploration and exploitation to maximize the score.
BEFORE giving the music prompts, you should spend time to reason on the classification and scores of the previous prompts, and understand what makes a prompt successful for the user, what makes it fail, how to combine the acquired knowledge and where we are not exploring, for example if a music generne is not being explored or if the prompts are too similar.
You should also try to understand and reason about the user preferences based on the scores and the classification of the prompts, and how to exploit this knowledge to generate better prompts.
AFTER this careful reasoning about the current evaluation, you should generate a diverse set of prompts that are likely to be successful tying to not repeat te same patterns and content in the requested format.

Here is the current population with their similarity scores and ranking for the current generation:
{ranking}

after the reasonin, generate only the next generation of prompts with a population of {self._population_size-len(new_prompts)} prompts."""

            answers = self.LLM_model.query_llm(LLM_prompt)
            generated_prompts = self.LLM_model.parse_llm_response(answers)
            new_prompts += generated_prompts[: self._population_size - len(new_prompts)]
            
        # print("Current Population:\n\t- ", "\n\t- ".join(self._problem.prompts))
        # print("New Population:\n\t- ", "\n\t- ".join(new_prompts))
        print("Finished generating new prompts.")

        # Update the population
        self._problem.prompts = new_prompts

    def _step(self):
        """Perform a step of the solver"""
        # Evaluate the population
        self._problem.evaluate(self._population)
        
        if self.config.mode == "full LLM":
            self.full_LLM_step()
        elif self.config.mode == "LLM evolve":
            raise NotImplementedError("LLM evolve mode is not yet implemented")
        else:
            raise ValueError("Invalid search mode")

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
            initial_bounds=(-1, 1),
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

        if self.prompt_optim:
            # Prompt Optimization Mode
            index = int(solution.values.item())
            audio_path = self.music_generator.generate_music(
                input=self.prompts[index], name=f"music_intermediate"
            )
        else:
            # Embedding Optimization Mode
            embedding = solution.values
            # copy to not be read only
            embedding = embedding.clone().detach()
            audio_path = self.music_generator.generate_music(input=embedding, name=f"music_intermediate")

        # Compute the embedding of the generated music
        fitness = self.evaluator.compute_fitness([audio_path]).suqeeze()

        # Clean up generated audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            # print(f"Deleted temporary audio file: {audio_path}")

        print(f"Generated: {self.generated} / {self.population_size}")
        if self.generated >= self.population_size:
            self.generated = 0
            print("Finished generation for this population.")

        del generated_embedding
        torch.cuda.empty_cache()  # Clears the GPU cache
        gc.collect()  # Forces Python to perform garbage collection

        solution.set_evals(fitness)

    def _fill(self, values: torch.Tensor):
        prompts = []
        population = values.shape[0]

        print(
            f"Generating diverse prompts for the initial population of {population} solutions..."
        )

        # Generate diverse prompts for the initial population
        prompts = self.LLM_model.generate_prompts(population)

        processed_prompts = self.music_generator.preprocess_text(prompts)
        
        if self.text_mode:
            values.copy_(processed_prompts)
        else:
            values.copy_(processed_prompts.view(population, -1))
            
        return values

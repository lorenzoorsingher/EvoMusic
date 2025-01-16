import os

from sklearn.decomposition import PCA
import joblib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from evotorch.logging import Logger
from evotorch import Problem
from evotorch.algorithms import SearchAlgorithm
import wandb

from EvoMusic.configuration import evolutionLogger
from music_generation.generators import MusicGenerator
from EvoMusic.evolution.searchers import MusicOptimizationProblem

class LivePlotter(Logger):
    def __init__(
        self,
        searcher: SearchAlgorithm,
        problem: MusicOptimizationProblem,
        music_generator: MusicGenerator,
        config: dict,
        logger_config: evolutionLogger,
    ):
        # Call the super constructor
        super().__init__(searcher)

        self.config = logger_config
        self.generator = music_generator
        
        self.searcher = searcher
        self.problem = problem

        if self.config.wandb:
            wandb.init(
                project=self.config.project, 
                name=self.config.name, 
                config=config
            )

        # Initialize data containers
        self.iterations = []
        self.fitness_values = []
        self.best_fitness_history = []

        self.best_embedding_history = []

        if self.config.visualizations:
            matplotlib.use("TkAgg")

            # Create a figure with three subplots: 2D Evolution, 3D Embeddings, and 2D Embeddings
            self._fig = plt.figure(
                figsize=(15, 7), dpi=100
            )  # Increased width to accommodate an extra subplot

            # 2D Plot for Iteration vs. Fitness
            self._ax2D = self._fig.add_subplot(1, 1, 1)
            self._ax2D.set_xlabel("Iteration")
            self._ax2D.set_ylabel("fitness")
            self._ax2D.set_title("Evolution Progress")

            # Legends for both plots
            self._ax2D.legend(loc="upper right")

            # Set interactive mode on
            plt.ion()
            plt.show()

    def _log(self, status: dict):
        # Update iteration and fitness history
        current_iter = status["iter"]
        self.iterations.append(current_iter)

        # Update best fitness history
        best_fitness = status["pop_best_eval"]
        best = status["pop_best"].values
        self.best_fitness_history.append(best_fitness)

        if self.config.visualizations:
            # Update 2D Evolution Progress Plot
            self._ax2D.clear()
            self._ax2D.plot(
                self.iterations,
                self.best_fitness_history,
                label="Best Fitness",
                color="green",
            )
            self._ax2D.set_xlabel("Iteration")
            self._ax2D.set_ylabel("fitness")
            self._ax2D.set_title("Evolution Progress")
            self._ax2D.legend(loc="upper right")
            self._ax2D.grid(True)
            
            # Draw the plot
            plt.draw()

        if self.config.wandb:
            wandb.log({"Best Fitness": best_fitness}, step=current_iter)
            
            if self.problem.text_mode:
                # get all the prompts
                prompts = self.searcher.population.values
                # convert ObjectArray to string list
                prompts = [prompt for prompt in prompts]
                # get all the evaluations
                evals = self.searcher.population.evals.view(-1).cpu().numpy()
                
                # log the prompts and evaluations
                table = wandb.Table(columns=["Prompt", "Fitness"])
                for prompt, fitness in zip(prompts, evals):
                    table.add_data(prompt, fitness)
                wandb.log({"Prompts Table": table}, step=current_iter)
                
        if self.problem.text_mode:
            print(f"Iteration: {current_iter} | Best Fitness: {best_fitness} | Best Prompt: {best}")
        else:   
            print(f"Iteration: {current_iter} | Best Fitness: {best_fitness}")
        
        best_audio_path = self.generator.generate_music(
            input=best, 
            name="BestPop" + str(current_iter), 
            duration=self.problem.evo_config.duration
        )
        
        if self.config.wandb:
            wandb.log({"Best Audio": wandb.Audio(best_audio_path, caption=best)}, step=current_iter)
        


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

if __name__ == "__main__":
    import sys

    sys.path.append("./")
    sys.path.append("../")

from configuration import evolutionLogger
from music_generation.generators import MusicGenerator
from evolution.searchers import MusicOptimizationProblem

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

        if self.config.wandb:
            wandb.init(
                project=self.config.project, 
                name=self.config.name, 
                config=config
            )

        self._searcher = searcher
        self._problem = problem

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
        current_fitness = status[self._target_status]
        self.iterations.append(current_iter)
        self.fitness_values.append(current_fitness)

        # Update best fitness history
        best_fitness = max(self.fitness_values)
        self.best_fitness_history.append(best_fitness)

        if self.config.visualizations:
            # Update 2D Evolution Progress Plot
            self._ax2D.clear()
            self._ax2D.plot(
                self.iterations,
                self.fitness_values,
                label="Current Fitness",
                color="blue",
            )
            self._ax2D.plot(
                self.iterations,
                self.best_fitness_history,
                label="Best Fitness",
                color="green",
            )
            self._ax2D.set_xlabel("Iteration")
            self._ax2D.set_ylabel(self._target_status)
            self._ax2D.set_title("Evolution Progress")
            self._ax2D.legend(loc="upper right")
            self._ax2D.grid(True)

        if self.config.wandb:
            wandb.log({self._target_status: current_fitness}, step=current_iter)
            wandb.log({"Best Fitness": best_fitness}, step=current_iter)

        print(f"Iteration: {current_iter} | {self._target_status}: {current_fitness}")
        breakpoint()
        best = status["pop_best"].values
        self.generator.generate_music(
            input=best, 
            name="BestPop" + str(current_iter), 
            duration=self._problem.evo_config.duration
        )


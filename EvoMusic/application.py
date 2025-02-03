import os
from diffusers.utils.testing_utils import enable_full_determinism
from tqdm import tqdm
import wandb
from EvoMusic.evolution.evolve import MusicEvolver
from EvoMusic.configuration import load_yaml_config
from EvoMusic.music_generation.generators import EasyRiffPipeline, MusicGenPipeline
from EvoMusic.usrapprox.utils.user_train_manager import UsersTrainManager
from EvoMusic.usrapprox.utils.user import RealUser, SynthUser, User

import torch
import numpy as np
import random


class EvoMusic:
    def __init__(self, config_path: str):
        self.config = load_yaml_config(config_path)

        if self.config.music_model == "riffusion":
            self.music_generator = EasyRiffPipeline(self.config.riffusion_pipeline)
        elif self.config.music_model == "musicgen":
            self.music_generator = MusicGenPipeline(self.config.music_generator)
        else:
            raise ValueError(
                "Invalid music model specified. Choose between 'musicgen' and 'riffusion'."
            )

        self.user_mapping = [0] * self.config.user_model.user_conf.amount
        if self.config.evolution.fitness.mode in ["user", "dynamic"]:
            users = []
            for i, user_conf in enumerate(self.config.user_model.users):
                type = user_conf.user_type
                if type == "real":
                    user = RealUser(1000 + i)
                    self.user_mapping[i] = 1000 + i
                elif type == "synth":
                    user = SynthUser(user_conf.target_user_id)
                    self.user_mapping[i] = user_conf.target_user_id
                else:
                    raise ValueError(
                        "Invalid user type specified. Choose between 'real' and 'synth'."
                    )
                users.append(user)
            self.epoch = [0] * self.config.user_model.user_conf.amount

            self.user_manager = UsersTrainManager(
                users=users,
                users_config=self.config.user_model.user_conf,
                aligner_config=self.config.user_model.aligner,
                train_config=self.config.user_model.train_conf,
                device=self.config.user_model.device,
            )
        elif self.config.evolution.fitness.mode == "music":
            self.user_manager = None
            self.epoch = [0]
        else:
            raise ValueError(
                "Invalid search mode specified. Choose between 'music', 'user', and 'dynamic'."
            )

        self.evolver = MusicEvolver(self.config.evolution, self.music_generator)

    def __get_user_fitness(self, user_idx: int):
        user = self.user_manager.get_user(self.user_mapping[user_idx])

        def user_fitness(solution):
            solution = solution.to(self.config.user_model.device)
            if self.config.evolution.fitness.mode == "dynamic":
                _, _, _, score = self.user_manager.get_user_score(user, solution)
            else:
                _, _, _, score = self.user_manager.get_reference_score(user, solution)
            return score

        return user_fitness

    def evolve(self, user_idx: int = 0, n_generations: int = None):
        """
        Evolve prompts or embeddings.

        Args:
            user_idx (int, optional): Index of the user to evolve. Defaults to 0.
            n_generations (int, optional): Number of generations to evolve. Defaults to None (uses config value).

        Returns:
            dict: Results of the evolution process.
        """
        if self.config.evolution.fitness.mode != "music":
            assert (
                user_idx < self.config.user_model.user_conf.amount
            ), "Invalid user index."
            print(f"[APP] Evolving user {user_idx}...")

            # Evolve the user
            user_fitness = self.__get_user_fitness(user_idx)
        else:
            # the internal fitness already computes the similarity with the reference
            user_fitness = None

        results = self.evolver.evolve(
            n_generations=n_generations, user_fitness=user_fitness
        )
        return results

    def generate_music(self, results, base_name: str, duration: int = None):
        """
        Generate music from the best solution.

        Args:
            results (dict): Results of the evolution process.
            base_name (str): Base name of the audio file.
            duration (int, optional): Duration of the generated music. Defaults to None (uses config value).
        """
        if duration is None:
            duration = self.config.evolution.best_duration

        print(f"[APP] Generating music...")
        audio_paths = []

        solutions = results["last_generation"]["solutions"]
        fitnesses = results["last_generation"]["fitness_values"]
        # Sort solutions by fitness (higher is better)
        solutions, fitnesses = zip(
            *sorted(zip(solutions, fitnesses), key=lambda x: x[1], reverse=True)
        )
        
        if self.config.user_model.sample:
            # use random sampling based on rank
            prob = [1 / (i + 1) for i in range(len(solutions))]
            sum_prob = sum(prob)
            prob = [p / sum_prob for p in prob]
            indices = np.random.choice(len(solutions), self.config.user_model.best_solutions, replace=False, p=prob)
            solutions = [solutions[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]
        else:
            solutions = solutions[: self.config.user_model.best_solutions]
            fitnesses = fitnesses[: self.config.user_model.best_solutions]

        for i, fit, sol in zip(range(len(solutions)), fitnesses, solutions):
            if not self.evolver.problem.text_mode:
                # copy the input to a new tensor as the values are read-only
                sol = sol.clone().detach()
                if self.music_generator.config.input_type == "token_embeddings":
                    print(f"\t{i+1}. {self.music_generator.token_to_text(sol)} ~ {fit}")
                else:
                    print(f"\t{i+1}. embedding with fitness score {fit}")
            else:
                print(f"\t{i+1}. {sol} - {fit}")

        for i, solution in tqdm(enumerate(solutions), total=len(solutions)):
            if not self.evolver.problem.text_mode:
                generator_input = solution.clone().detach()
            else:
                generator_input = solution
            audio_path = self.music_generator.generate_music(
                input=generator_input, name=f"{base_name}_{i}", duration=duration
            )
            audio_paths.append(audio_path)

        return audio_paths

    def finetune_user(self, songs, user_idx=0):
        if self.config.evolution.fitness.mode != "dynamic":
            self.epoch[user_idx] += 1
            return

        # compute true user likeness using reference and log it
        print(f"[APP] Computing user likeness for user {user_idx}...")
        user = self.user_manager.get_user(self.user_mapping[user_idx])
        batch = self.evolver.problem.evaluator.embed_audios(songs).unsqueeze(0)

        print(f"[APP] Finetuning user {user_idx}...")
        # if user is a real user, then send the list of songs to it
        # breakpoint()
        if isinstance(user, RealUser):
            user.set_playlist(songs)
            
        _, _, _, score = self.user_manager.get_reference_score(user, batch)
        # log max, min, mean, std of the scores
        wandb.log(
            {f"user: {user_idx} Validation/max score generated": score.max().item()}
        )
        wandb.log(
            {f"user: {user_idx} Validation/min score generated": score.min().item()}
        )
        wandb.log(
            {f"user: {user_idx} Validation/mean score generated": score.mean().item()}
        )
        wandb.log(
            {f"user: {user_idx} Validation/std score generated": score.std().item()}
        )
        wandb.log(
            {
                f"user: {user_idx} Validation/median score generated": score.median().item()
            }
        )

        self.user_manager.finetune(user, batch, self.epoch[user_idx])
        self.epoch[user_idx] += 1

    def generation_loop(
        self, user_idx: int = 0, n_generations: int = None, epochs: int = None
    ):
        """
        Evolve prompts or embeddings then finetune the user, rinse and repeat.

        Args:
            user_idx (int): Index of the user to evolve.
            n_generations (int, optional): Number of generations to evolve. Defaults to None (uses config value).
            epochs (int, optional): Number of epochs to run. Defaults to None (uses config value).
        """
        if epochs is None:
            epochs = self.config.epochs
        
        for i in range(epochs):
            if self.config.evolution.fitness.mode == "dynamic" and i == 0:
                self.single_step(user_idx, 1) # first step is to generate the initial music and get the initial feedback to finetune the user
            else:
                self.single_step(user_idx, n_generations)

    def single_step(self, user_idx: int = 0, n_generations: int = None):
        """
        Perform a single step of evolution and finetuning.

        Args:
            user_idx (int): Index of the user to evolve.
            n_generations (int, optional): Number of generations to evolve. Defaults to None (uses config
        """
        print(
            "\n\n====================================================================================================="
        )
        print(
            f"Performing a single step of evolution and finetuning for user {user_idx} epoch {self.epoch[user_idx]}"
        )
        print(
            "====================================================================================================="
        )

        results = self.evolve(user_idx, n_generations)
        if not os.path.exists(
            f"{self.music_generator.config.output_dir}/{self.config.evolution.exp_name}_music_{user_idx}_epoch_{self.epoch[user_idx]}"
        ):
            os.makedirs(
                f"{self.music_generator.config.output_dir}/{self.config.evolution.exp_name}_music_{user_idx}_epoch_{self.epoch[user_idx]}"
            )
        music = self.generate_music(
            results,
            f"{self.config.evolution.exp_name}_music_{user_idx}_epoch_{self.epoch[user_idx]}/",
            self.config.evolution.best_duration,
        )
        self.finetune_user(music, user_idx)

        print(
            "==========================================================================================="
        )
        print(
            f"Evolution and finetuning for user {user_idx} completed for epoch {self.epoch[user_idx]-1}"
        )
        print(
            "==========================================================================================="
        )

        return music

    def start(self):
        """
        Start the evolution process with an UI. (users only)
        """
        import gradio as gr

        def run_evolution(user_index, n_generations):
            music = self.single_step(int(user_index), int(n_generations))
            return "Evolution completed", music

        gr.Interface(
            fn=run_evolution,
            inputs=[
                gr.Number(label="User Index", value=0),
                gr.Number(label="Number of Generations", value=1),
            ],
            outputs=[
                gr.Text(label="Status"),
                gr.File(label="Generated Songs", file_count="multiple"),
            ],
            title="Evolution UI",
        ).launch()


if __name__ == "__main__":
    enable_full_determinism()

    # Set random seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.seed_all()

    evo_music = EvoMusic("config.yaml")
    evo_music.generation_loop(0)

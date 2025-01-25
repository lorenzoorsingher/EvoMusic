from diffusers.utils.testing_utils import enable_full_determinism
from EvoMusic.evolution.evolve import MusicEvolver
from EvoMusic.configuration import load_yaml_config
from EvoMusic.music_generation.generators import EasyRiffPipeline, MusicGenPipeline
from EvoMusic.usrapprox.utils.user_train_manager import UsersTrainManager
from EvoMusic.usrapprox.utils.user import RealUser, SynthUser, User

import torch
from torch.utils.tensorboard import SummaryWriter
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
                    user = RealUser(1000+i)
                    self.user_mapping[i] = 1000+i
                elif type == "synth":
                    user = SynthUser(user_conf.target_user_id)
                    self.user_mapping[i] = user_conf.target_user_id
                else:
                    raise ValueError(
                        "Invalid user type specified. Choose between 'real' and 'synth'."
                    )
                users.append(user)
            self.epoch = [0] * self.config.user_model.user_conf.amount
            
            writer = SummaryWriter()
            self.user_manager = UsersTrainManager(
                users = users,
                users_config = self.config.user_model.user_conf,
                aligner_config=self.config.user_model.aligner,
                train_config = None,
                writer=writer,
                device = self.config.user_model.device,
            )
        elif self.config.evolution.fitness.mode == "music":
            self.user_manager = None
        else:
            raise ValueError(
                "Invalid search mode specified. Choose between 'music', 'user', and 'dynamic'."
            )
            
        self.evolver = MusicEvolver(self.config.evolution, self.music_generator)
        
    def __get_user_fitness(self, user_idx: int):
        user = self.user_manager.get_user(self.user_mapping[user_idx])
        def user_fitness(solution):
            if self.config.evolution.search.mode == "user":
                _,_,_, score = self.user_manager.get_user_score(user, solution)
            else:
                _,_,_, score = self.user_manager.get_reference_score(user, solution)
            return score
        return user_fitness
        
    def evolve(self, user_idx: int = None, n_generations: int = None):
        """
        Evolve prompts or embeddings.
        
        Args:
            user_idx (int, optional): Index of the user to evolve. Defaults to 0.
            n_generations (int, optional): Number of generations to evolve. Defaults to None (uses config value).
            
        Returns:
            dict: Results of the evolution process.
        """
        if self.config.evolution.fitness.mode != "music":
            # Check input validity
            if user_idx is None:
                user_idx = 0
            assert user_idx < self.config.user_model.user_conf.amount, "Invalid user index."
            print(f"[APP] Evolving user {user_idx}...")
            
            # Evolve the user
            user_fitness = self.__get_user_fitness(user_idx)
        else:
            user_fitness = None
        
        results = self.evolver.evolve(n_generations=n_generations, user_fitness=user_fitness)
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
        
        audio_paths = []
        for i, solution in enumerate(results["last_generation"]["solutions"][:10]):
               
            if not self.evolver.problem.text_mode:
                # copy the input to a new tensor as the values are read-only
                generator_input = solution.clone().detach()
            else:
                generator_input = solution
                
            audio_path = self.music_generator.generate_music(
                input=generator_input, 
                name=f"{base_name}_{i}", 
                duration=duration
                )
            audio_paths.append(audio_path)
            
        return audio_paths
    
    def finetune_user(self, songs, user_idx):
        print(f"[APP] Finetuning user {user_idx}...")
        batch = self.evolver.problem.evaluator.embed_audios(songs).unsqueeze(0)
        self.user_manager.finetune(
            self.user_manager.get_user(self.user_mapping[user_idx]), 
            batch, self.epoch[user_idx])
        self.epoch[user_idx] += 1

    def generation_loop(self, user_idx: int):
        """
        Evolve prompts or embeddings then finetune the user, rinse and repeat.
        
        Args:
            user_idx (int): Index of the user to evolve.
        """
        while True:
            results = self.evolve(user_idx)
            music = self.generate_music(results, f"music_{user_idx}/{self.epoch[user_idx]}")
            self.finetune_user(music, user_idx)

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
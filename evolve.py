import os
import evotorch
import torch
import torchaudio
from evotorch import Problem, Solution
from evotorch.algorithms import CMAES, SearchAlgorithm
from evotorch.algorithms.searchalgorithm import SinglePopulationAlgorithmMixin
from diffusers import DiffusionPipeline
from diffusers.utils.testing_utils import enable_full_determinism
from sklearn.decomposition import PCA
import joblib

from transformers import AutoModel, Wav2Vec2FeatureExtractor

import requests
import json
import math
import gradio as gr
from evotorch.logging import Logger
import matplotlib
import matplotlib.pyplot as plt
import random
import gc
import numpy as np
from tqdm.auto import tqdm
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

import sys

sys.path.append("./")

sys.path.append("music_generation")
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

# ----------------------------- Configuration -----------------------------

# Load environment variables
load_dotenv()

# Paths and IDs
TARGET_AUDIO_PATH = "generated_audio/breaking_me_down.mp3"  # <-- Replace with your actual target audio path
MODEL_ID = "riffusion/riffusion-model-v1"
# LLM_API_URL = "http://127.0.0.1:1234/v1/chat/completions"  # Ensure this is your running LLM API endpoint
# LLM_MODEL = "llama-3.2-1b-instruct"  # Ensure this matches your LLM model
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-4o-mini"
API_KEY = os.getenv("API_KEY")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 50
ELITES = 0.1
CROSS_MUTATION = 0
NOVEL_PROMPTS = 0.1
TOURNAMENT_SIZE = 5
NUM_GENERATIONS = 100


NUM_INFERENCE_STEPS = 50
DURATION = 5  # in seconds
HOP_SIZE = 0.1

TEMPERATURE_GENERATION = 0.8
TEMPERATURE_EVOLVE = 0.5

# Optimization Mode
# Set to True for prompt optimization, False for embedding optimization
PROMPT_OPTIM = False
EMB_SIZE = 768 # Embedding size for CLIP model
MAX_SEQ_LEN = 1  # Maximum sequence length for CLIP model
USE_CMAES = True # Set to True to use CMA-ES instead of custom Searcher

# -------------------------------------------------------------------------

# ------------------------- Audio Embedding Setup -------------------------

# Load the tokenizer and model
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True).to(DEVICE)
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
resample_rate = processor.sampling_rate

def get_audio_embedding(audio_path):
    """
    Compute the embedding of an audio file using the EncodecModel.
    Ensures that the audio is stereo (2 channels) before processing.
    """
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    # print(f"Loaded audio '{audio_path}' with shape {waveform.shape} and sample rate {sample_rate} Hz.")

    # Resample audio if necessary
    if sample_rate != resample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)
        sample_rate = resample_rate

    # make audio mono if stereo
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze(0)

    with torch.no_grad():
        inputs = processor(waveform, sampling_rate=resample_rate, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        output = model(**inputs)
        embedding = output.last_hidden_state.mean(dim=1).squeeze(0)

    # Clear memory
    # del emb, ts, 
    del waveform  
    
    return embedding.to(DEVICE)

# compute PCA on embeddings using the songs in a music dataset
def compute_pca():
    audio_path = "generated_audio/songs"
    embeddings = []
    audios = os.listdir(audio_path)

    pca = PCA(n_components=min(768,len(audios)))
    pca_3D = PCA(n_components=3)

    # look if already computed
    if os.path.isfile("pca_3D.pkl") and os.path.isfile("pca.pkl"):
        pca = joblib.load("pca.pkl")
        pca_3D = joblib.load("pca_3D.pkl")
    else:
        for file in tqdm(audios):
            emb = get_audio_embedding(os.path.join(audio_path, file))
            embeddings.append(emb.cpu().numpy())
        embeddings = np.array(embeddings)

        pca.fit(embeddings)
        pca_3D.fit(embeddings)
        joblib.dump(pca, "pca.pkl")
        joblib.dump(pca_3D, "pca_3D.pkl")

        # show 3D PCA space
        embeddings = pca_3D.transform(embeddings)
        # normalize each embedding
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:,None]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embeddings[:,0], embeddings[:,1], embeddings[:,2])
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_zlabel('PCA 3')
        plt.show()

    return pca, pca_3D  # Return both PCA objects

pca, pca_3D = compute_pca()

# Compute target embedding
print("Computing target audio embedding...")
if not os.path.isfile(TARGET_AUDIO_PATH):
    raise FileNotFoundError(f"Target audio file not found at: {TARGET_AUDIO_PATH}")

target_embedding_full = get_audio_embedding(TARGET_AUDIO_PATH).to(DEVICE)
target_embedding = pca.transform(target_embedding_full.cpu().numpy().reshape(1,-1))[0]
target_embedding = torch.tensor(target_embedding).to(DEVICE)

# Project target embedding into 3D PCA space
target_embedding_3D = pca_3D.transform(target_embedding_full.cpu().numpy().reshape(1,-1))[0]
# normalize the embedding
target_embedding_3D = target_embedding_3D / np.linalg.norm(target_embedding_3D)
print("Target embedding computed.")

# ------------------------- Music Generation Setup ------------------------

# Load the Riffusion model pipeline
print("Loading Riffusion model pipeline...")
pipe = DiffusionPipeline.from_pretrained(MODEL_ID)
pipe = pipe.to(DEVICE)

# Dummy safety checker to bypass the safety check
def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)
pipe.safety_checker = dummy_safety_checker
# enable full deterministic mode
enable_full_determinism()

generator = torch.Generator(device=DEVICE)
print("Riffusion pipeline loaded.")

def generate_music_riffusion(prompt: str=None, embeds=None, num_inference_steps: int = NUM_INFERENCE_STEPS, duration: int = DURATION, name = None):
    """
    Generate music using the Riffusion model based on a text prompt or embeddings.
    Returns the path to the generated audio file.
    """

    assert (prompt is not None) or (embeds is not None), "Either prompt or embeds must be provided."

    width = math.ceil(duration * (512 / 5))  # Calculate the width based on the duration
    # must be divisible by 8
    width = width + (8 - width % 8) if width % 8 != 0 else width
    if prompt is not None:
        # print(f"Generating music with prompt: '{prompt}'")
        generator.manual_seed(0)
        output = pipe(prompt, num_inference_steps=num_inference_steps, width=width, generator=generator)
    else:
        # print(f"Generating music with embeddings.")
        embeds = embeds.view(1, MAX_SEQ_LEN, EMB_SIZE).to(DEVICE)
        generator.manual_seed(0)
        output = pipe(prompt_embeds=embeds, num_inference_steps=num_inference_steps, width=width, generator=generator)

    image = output.images[0]

    # Convert spectrogram image back to audio
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params=params)

    segment = converter.audio_from_spectrogram_image( image, apply_filters=True, )

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    if (name is None): 
        audio_filename = f"generated_music_{torch.randint(0, int(1e6), (1,)).item()}.wav"
    else: 
        audio_filename = f"{name}.wav"
    audio_path = os.path.join(output_dir, audio_filename)
    segment.export(audio_path, format="wav")
    # print(f"Generated audio saved at: {audio_path}")

    del image, segment, output  # Clear memory
    return audio_path

generate_music = generate_music_riffusion

# ------------------------- LLM Query Function ----------------------------

def query_llm(prompt: str, temperature=0.5):
    """
    Query the LLM API with the given prompt.
    """
    # print(f"Querying LLM with prompt: '{prompt}'")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You produce prompts used to generate music following the requests of the user. You should always respond with only the requested prompts and by encasing each one of the produced prompts in <p> and </p> tags. Like the following: 1. <p> A music prompt. </p> 2. <p> Another music prompt. </p>"},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    try:
        response = requests.post(LLM_API_URL, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        llm_response = response.json()["choices"][0]["message"]["content"].strip()
        # print(f"LLM responded with: '{llm_response}'")
        return llm_response
    except Exception as e:
        print(f"LLM API request failed: {e}")
        return "A default music prompt."

# ------------------------- Evotorch Components ---------------------------

class PromptOptimizationProblem(Problem):
    """
    Evotorch Problem for optimizing music prompts or embeddings.
    """
    def __init__(self, target_embedding, target_embedding_3D, prompt_optim=PROMPT_OPTIM, population_size=POPULATION_SIZE):
        super().__init__(
            objective_sense="max",
            solution_length= 1 if prompt_optim else EMB_SIZE * MAX_SEQ_LEN,
            initial_bounds=(-1, 1),
            device="cpu"
        )
        self.target_embedding = target_embedding
        self.target_embedding_3D = target_embedding_3D
        self.population_size = population_size
        self.generated = 0
        self.embeddings_3D = []
        self.prompt_optim = prompt_optim
        self.device_problem = "cpu"

    def _evaluate(self, solution: Solution):
        """
        Objective function that maps solution vectors to prompts or embeddings, generates music,
        computes embeddings, and evaluates similarity to the target embedding.
        """
        self.generated += 1

        if self.prompt_optim:
            # Prompt Optimization Mode
            index = int(solution.values.item())
            audio_path = generate_music(prompt=self.prompts[index])
        else:
            # Embedding Optimization Mode
            embedding = solution.values
            audio_path = generate_music(embeds=embedding)

        # Compute the embedding of the generated music
        generated_embedding_full = get_audio_embedding(audio_path)
        generated_embedding = pca.transform(generated_embedding_full.cpu().numpy().reshape(1, -1))[0]
        generated_embedding = torch.tensor(generated_embedding).to(DEVICE)

        # Project into 3D PCA space
        generated_embedding_3D = pca_3D.transform(generated_embedding_full.cpu().numpy().reshape(1, -1))[0]
        # normalize the embedding
        generated_embedding_3D = generated_embedding_3D / np.linalg.norm(generated_embedding_3D)
        # Store the 3D embedding for visualization
        self.embeddings_3D.append(generated_embedding_3D)

        similarity = torch.cosine_similarity(self.target_embedding, generated_embedding, dim=0).item()

        # print(f"Vector similarity: {similarity}")
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

        solution.set_evals(similarity)

    def _fill(self, values: torch.Tensor):
        prompts = []
        population = values.shape[0]

        prompt_length = ""
        if not self.prompt_optim:
            prompt_length = f" of maximum {MAX_SEQ_LEN} words each"

        while len(prompts) < population:
            answers = query_llm(f"Generate {population-len(prompts)} diverse prompts for generating music{prompt_length} you should generate higly diverse prompts both in structure and contents, spanning different music styles, instruments, emotions, ...", temperature=TEMPERATURE_GENERATION)

            # match the number of <p> and </p> tags if not equal then ignore the prompt
            if answers.count("<p>") != answers.count("</p>"):
                continue

            for answer in answers.split("</p>"):
                if "<p>" in answer:
                    prompts.append(answer[answer.index("<p>")+3:].strip())
                if len(prompts) >= population:
                    break

        if self.prompt_optim:
            self.prompts = prompts
            values.copy_(torch.tensor([i for i in range(population)]).view(-1, 1))
        else:
            # use pipe text_encoder to encode the prompts
            with torch.no_grad():
                outputs = pipe.encode_prompt(prompts, device=DEVICE, num_images_per_prompt=1, do_classifier_free_guidance=True)
                embeddings = outputs[0]
            values.copy_(embeddings.view(population, -1)[:, :EMB_SIZE * MAX_SEQ_LEN])
        
        # del outputs, embeddings
        return values

matplotlib.use("TkAgg")
class LivePlotter(Logger):
    def __init__(self, searcher, problem, target_status: str):
        # Call the super constructor
        super().__init__(searcher)
        self._searcher = searcher
        self._problem = problem
        self._target_embedding_3D = problem.target_embedding_3D
        self._target_embedding_2D = problem.target_embedding_3D[:2] / np.linalg.norm(problem.target_embedding_3D[:2])

        # Set up the target status
        self._target_status = target_status

        # Create a figure with three subplots: 2D Evolution, 3D Embeddings, and 2D Embeddings
        self._fig = plt.figure(figsize=(15, 7), dpi=100)  # Increased width to accommodate an extra subplot

        # 2D Plot for Iteration vs. Fitness
        self._ax2D = self._fig.add_subplot(1, 3, 1)
        self._ax2D.set_xlabel("Iteration")
        self._ax2D.set_ylabel(target_status)
        self._ax2D.set_title("Evolution Progress")

        # 3D Plot for Embeddings
        self._ax3D = self._fig.add_subplot(1, 3, 2, projection='3d')
        self._ax3D.set_xlabel("PCA 1")
        self._ax3D.set_ylabel("PCA 2")
        self._ax3D.set_zlabel("PCA 3")
        self._ax3D.set_title("3D Embedding Space")

        # NEW: 2D Plot for Embeddings (PCA1 vs PCA2)
        self._ax2D_emb = self._fig.add_subplot(1, 3, 3)
        self._ax2D_emb.set_xlabel("PCA 1")
        self._ax2D_emb.set_ylabel("PCA 2")
        self._ax2D_emb.set_title("2D Embedding Space (PCA1 vs PCA2)")

        # Initialize data containers
        self.iterations = []
        self.fitness_values = []
        self.best_fitness_history = []

        self.best_embedding_history = []

        # Plot elements for 3D Embedding Space
        self.current_population_scatter = self._ax3D.scatter([], [], [], c='blue', label='Current Population', alpha=0.6)
        self.best_scatter = self._ax3D.scatter([], [], [], c='green', marker='*', s=100, label='Best Solution')
        self.past_bests_scatter = self._ax3D.scatter([], [], [], c='orange', marker='D', s=50, label='Past Bests')
        self.target_scatter = self._ax3D.scatter(
            target_embedding_3D[0],
            target_embedding_3D[1],
            target_embedding_3D[2],
            c='red',
            marker='X',
            s=150,
            label='Target'
        )

        # NEW: Plot elements for 2D Embedding Space
        self.current_population_scatter_2D = self._ax2D_emb.scatter([], [], c='blue', label='Current Population', alpha=0.6)
        self.best_scatter_2D = self._ax2D_emb.scatter([], [], c='green', marker='*', s=100, label='Best Solution')
        self.past_bests_scatter_2D = self._ax2D_emb.scatter([], [], c='orange', marker='D', s=50, label='Past Bests')
        self.target_scatter_2D = self._ax2D_emb.scatter(
            self._target_embedding_2D[0],
            self._target_embedding_2D[1],
            c='red',
            marker='X',
            s=150,
            label='Target'
        )

        # Legends for both plots
        self._ax2D.legend(loc='upper right')
        self._ax3D.legend(loc='upper left')
        self._ax2D_emb.legend(loc='upper left')  # Add legend for the new 2D plot

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

        # Update 2D Evolution Progress Plot
        self._ax2D.clear()
        self._ax2D.plot(self.iterations, self.fitness_values, label='Current Fitness', color='blue')
        self._ax2D.plot(self.iterations, self.best_fitness_history, label='Best Fitness', color='green')
        self._ax2D.set_xlabel("Iteration")
        self._ax2D.set_ylabel(self._target_status)
        self._ax2D.set_title("Evolution Progress")
        self._ax2D.legend(loc='upper right')
        self._ax2D.grid(True)

        # Update 3D Embedding Space Plot
        # Clear previous current population scatter
        if hasattr(self, 'current_population_scatter'):
            self.current_population_scatter.remove()

        # Extract current population embeddings
        current_population_embeddings = np.array([
            self._problem.embeddings_3D[i] for i in range(len(self._problem.embeddings_3D))
        ])

        # Scatter current population
        self.current_population_scatter = self._ax3D.scatter(
            current_population_embeddings[:, 0],
            current_population_embeddings[:, 1],
            current_population_embeddings[:, 2],
            c='blue',
            label='Current Population',
            alpha=0.6
        )

        # Update best solution
        best_idx = self._searcher._population.evals.argmax()
        best_embedding = self._problem.embeddings_3D[best_idx]

        # Clear previous best and past best scatters
        if hasattr(self, 'best_scatter'):
            self.best_scatter.remove()
        if hasattr(self, 'past_bests_scatter') and len(self.best_embedding_history) > 1:
            self.past_bests_scatter.remove()

        # Plot best solution
        self.best_scatter = self._ax3D.scatter(
            best_embedding[0],
            best_embedding[1],
            best_embedding[2],
            c='green',
            marker='*',
            s=100,
            label='Best Solution'
        )

        # Plot past bests
        if len(self.best_embedding_history) > 0:
            history = np.array(self.best_embedding_history)
            self.past_bests_scatter = self._ax3D.scatter(
                history[:, 0],
                history[:, 1],
                history[:, 2],
                c='orange',
                marker='D',
                s=50,
                label='Past Bests'
            )
        self.best_embedding_history.append(best_embedding)
        
        # Re-add target scatter (it doesn't change)
        self.target_scatter.remove()
        self.target_scatter = self._ax3D.scatter(
            self._target_embedding_3D[0],
            self._target_embedding_3D[1],
            self._target_embedding_3D[2],
            c='red',
            marker='X',
            s=150,
            label='Target'
        )

        # Adjust the view
        self._ax3D.view_init(elev=30, azim=45)

        # Add legend only once
        handles, labels = self._ax3D.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self._ax3D.legend(by_label.values(), by_label.keys())

        # Center in 0,0,0
        self._ax3D.set_xlim(-1, 1)
        self._ax3D.set_ylim(-1, 1)
        self._ax3D.set_zlim(-1, 1)

        # Update 2D Embedding Space Plot
        current_population_embeddings = current_population_embeddings[:, :2]
        current_population_embeddings = current_population_embeddings / np.linalg.norm(current_population_embeddings, axis=1)[:, None]
        self._ax2D_emb.clear()
        # Current Population
        self._ax2D_emb.scatter(
            current_population_embeddings[:, 0],
            current_population_embeddings[:, 1],
            c='blue',
            label='Current Population',
            alpha=0.6
        )    
        past_bests = np.array([emb[:2] for emb in self.best_embedding_history])
        past_bests = past_bests / np.linalg.norm(past_bests, axis=1)[:, None]
        # Best Solution
        self._ax2D_emb.scatter(
            past_bests[-1, 0],
            past_bests[-1, 1],
            c='green',
            marker='*',
            s=100,
            label='Best Solution'
        )
        # Past Bests
        if len(self.best_embedding_history) > 1:
            self._ax2D_emb.scatter(
                past_bests[:-1, 0],
                past_bests[:-1, 1],
                c='orange',
                marker='D',
                s=50,
                label='Past Bests'
            )
        # Target Embedding
        self._ax2D_emb.scatter(
            self._target_embedding_2D[0],
            self._target_embedding_2D[1],
            c='red',
            marker='X',
            s=150,
            label='Target'
        )
        self._ax2D_emb.set_xlabel("PCA 1")
        self._ax2D_emb.set_ylabel("PCA 2")
        self._ax2D_emb.set_title("2D Embedding Space (PCA1 vs PCA2)")
        self._ax2D_emb.legend(loc='upper left')
        self._ax2D_emb.grid(True)

        self._ax2D_emb.set_xlim(-1, 1)
        self._ax2D_emb.set_ylim(-1, 1)

        # Draw and pause briefly to update the plot
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.1)

        self._problem.embeddings_3D = []

        if self._problem.prompt_optim:
            best = self._searcher._get_best()
            generate_music(prompt=best, name="best_music")
        else:
            best = status["pop_best"].values
            generate_music(embeds=best, name="best_music")



# ------------------------- Evolutionary Algorithm ------------------------

class PromptSearcher(SearchAlgorithm, SinglePopulationAlgorithmMixin):
    def __init__(self, problem:Problem, elites=ELITES, cross_mutation=CROSS_MUTATION, tournament_size=TOURNAMENT_SIZE, novel_prompts = NOVEL_PROMPTS):
        SearchAlgorithm.__init__(self, problem)

        self._problem = problem
        self._population_size = problem.population_size
        self._population = problem.generate_batch(self._population_size)
        self._elites = elites
        self._cross_mutation = cross_mutation
        self._tournament_size = tournament_size
        self._novel_prompts = novel_prompts
        self.best = None
        self.old_gen = ""

        SinglePopulationAlgorithmMixin.__init__(self)

    @property
    def population(self):
        return self._population
    
    def _step(self):
        """Perform a step of the solver"""
        # Evaluate the population
        self._problem.evaluate(self._population)
        indices = self._population.argsort()

        ranking = ""
        for i in range(len(indices)):
            ranking += f"{i+1}. <p> {self._problem.prompts[indices[i]]} </p> - {self._population[indices[i]].evals.item()*50+50} / 100\n"

        print("Pop Best", indices[0], ":", 
              self._problem.prompts[indices[0]], 
              self._population[indices[i]].evals.item()*50+50)
        
        self.best = self._problem.prompts[indices[0]]

        # keep some solutions sampled from the population with probability proportional to their fitness rank
        num_elites = int(self._elites * self._population_size)
        new_prompts = [self._problem.prompts[indices[i]] for i in range(num_elites)]

        # use random perturbation to generate new solutions using crossover and mutation
        num_simple_mutations = int(self._cross_mutation * self._population_size)
        for i in range(num_simple_mutations):
            parents = [ random.choice(indices) for _ in range(self._tournament_size) ]

            # choose the best parents
            parents.sort(key=lambda x: self._population[x].evals.item())
            parent1 = self._problem.prompts[parents[0]]
            parent2 = self._problem.prompts[parents[1]]

            # crossover
            words1 = parent1.split()
            words2 = parent2.split()
            if len(words1) == 0 or len(words2) == 0:
                continue
            crossover_point = random.randint(0, min(len(words1), len(words2)) - 1)
            new_prompt = " ".join(words1[:crossover_point] + words2[crossover_point:])
            new_prompts.append(new_prompt)

        # novel prompts for exploration
        num_novel = int(self._population_size * self._novel_prompts)
        novel_prompts = []
        while len(novel_prompts) < num_novel:
            answers = query_llm(f"Generate only {num_novel - len(novel_prompts)} diverse prompts for generating music, they should span multiple generes, moods, ...", temperature=TEMPERATURE_GENERATION)

            # match the number of <p> and </p> tags if not equal then ignore the prompt
            if answers.count("<p>") != answers.count("</p>"):
                continue

            for answer in answers.split("</p>"):
                if "<p>" in answer:
                    novel_prompts.append(answer[answer.index("<p>")+3:].strip())
                if len(novel_prompts) >= num_novel:
                    break
            
        new_prompts += novel_prompts

        # Update the population using LLM by giving it the scores of each prompt and asking for new ones
        while len(new_prompts) < self._population_size:
            LLM_prompt = f"""
Generate ONLY {self._population_size-len(new_prompts)} creative and diverse music prompts for generating music based on the classification and scores of the previous prompts. 
You should balance exploration and exploitation to maximize the score.
Before outputting the prompts, you should reason on the classification and scores of the previous prompts, and try to understand what makes a prompt successful for the user, and what makes it fail.
Following this reasoning, you should generate a diverse set of prompts that are likely to be successful in the requested format and number.

The previous generation was:
{self.old_gen}

And here is the current population with their similarity scores and ranking for the current generation:
{ranking}
                """
            answers = query_llm(LLM_prompt, temperature=TEMPERATURE_EVOLVE)

            # match the number of <p> and </p> tags if not equal then ignore the prompt
            if answers.count("<p>") != answers.count("</p>"):
                continue

            for answer in answers.split("</p>"):
                if "<p>" in answer:
                    new_prompts.append(answer[answer.index("<p>")+3:].strip())
                if len(new_prompts) >= self._population_size:
                    break
            
        self.old_gen = ranking

        print("Current Population:\n\t- ", "\n\t- ".join(self._problem.prompts))
        print("New Population:\n\t- ", "\n\t- ".join(new_prompts))

        # Update the population
        self._problem.prompts = new_prompts

    def _get_best(self):
        return self.best

def evolve_prompts(problem, generations=NUM_GENERATIONS):
    """
    Evolve prompts or embeddings to maximize similarity to target embedding using Evotorch's CMA-ES or custom Searcher.
    """
    if problem.prompt_optim:
        # Initialize the custom PromptSearcher optimizer
        optimizer = PromptSearcher(problem)
    else:
        # Initialize the CMA-ES optimizer for embedding optimization
        if USE_CMAES:
            optimizer = CMAES(problem, stdev_init=0.1, popsize=POPULATION_SIZE)
        else:
            optimizer = evotorch.algorithms.PGPE(problem, popsize=POPULATION_SIZE, center_learning_rate=1, stdev_learning_rate=1, stdev_init=1)

    # Run the evolution strategy
    print("Starting evolution...")
    LivePlotter(optimizer, problem, "pop_best_eval")
    optimizer.run(num_generations=generations)
    
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

# ----------------------------- Main Execution ----------------------------

if __name__ == "__main__":
    # Define the problem
    problem = PromptOptimizationProblem(target_embedding, target_embedding_3D, prompt_optim=PROMPT_OPTIM)

    # Evolve prompts or embeddings
    best_solution, best_fitness = evolve_prompts(problem)

    # Generate final music using the best prompt or embedding
    print("\nGenerating final music with the best solution...")
    if PROMPT_OPTIM:
        final_audio_path = generate_music(prompt=best_solution, name="final_music")
    else:
        best_embedding = best_solution.unsqueeze(0)  # Add batch dimension
        final_audio_path = generate_music(embeds=best_embedding, name="final_music")
    print(f"Final audio saved at: {final_audio_path}")

    # Launch Gradio interface to listen to the generated music
    def listen_to_music(audio_path):
        return audio_path

    with gr.Blocks() as demo:
        gr.Markdown("## Best Generated Music 🎶")
        output_audio = gr.Audio(label="Generated Music", value=final_audio_path)
        demo.launch()

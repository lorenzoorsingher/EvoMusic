import math
import os
from typing import Optional, Union

import transformers
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch import nn
from diffusers import (DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, PNDMScheduler,
                       LMSDiscreteScheduler)
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer, CLIPFeatureExtractor, \
    MusicgenForConditionalGeneration

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from diffusers.utils.testing_utils import enable_full_determinism


RIFFUSION_MODEL_ID = "riffusion/riffusion-model-v1"
MUSICGEN_MODEL_ID = "facebook/musicgen-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


class EasyDiffuse:
    def __init__(self):
        super().__init__()

    def text_to_embed(self, text):
        raise NotImplementedError

    def token_embedding_to_embed(self, token_embedding):
        raise NotImplementedError


class EasyRiffPipeline(EasyDiffuse):
    def __init__(self):
        super().__init__()
        self.model = DiffusionPipeline.from_pretrained(RIFFUSION_MODEL_ID).to(DEVICE)


    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedded_text = self.model.text_encoder(inputs.input_ids.to(self.model.device))
        return embedded_text.last_hidden_state

    def token_embedding_to_embed(self, token_embedding):
        with torch.no_grad():
            encoded = self.model.text_encoder.text_model.encoder(token_embedding.to(self.model.device)).last_hidden_state
            return self.model.text_encoder.text_model.final_layer_norm(encoded)

    def text_to_embeddings_before_clip(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.model.text_encoder.text_model.embeddings(inputs.input_ids.to(self.model.device))

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MusicGenHelpers(EasyDiffuse):
    def __init__(self):
        super().__init__()
        self.model = transformers.MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL_ID)

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.model.get_encoder()(**inputs).last_hidden_state

    def token_to_embed(self, tokens):
        with torch.no_grad():
            return self.model.get_encoder()(**tokens).last_hidden_state


if __name__ == "__main__":
    riffusion_pipe = EasyRiffPipeline()
    enable_full_determinism()
    generator = torch.Generator(device=DEVICE)
    width = math.ceil(3 * (512 / 5))
    width = width + (8 - width % 8) if width % 8 != 0 else width
    generator.manual_seed(0)
    embedding = riffusion_pipe.text_to_embed("Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive.", 30)
    #output = dh(prompt_embeds=embedding_clip, num_inference_steps=50, width=width, generator=generator)
    # image = output.images[0]
    # embedding_pre = dh.text_to_embeddings_before_clip("It's Another tequila sunrise staring closely across the sky", 15)
    # embedding_pre = dh.text_to_embeddings_before_clip("Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive.", 30)
    # embedding = dh.token_embedding_to_embed(embedding_pre)
    out2 = riffusion_pipe(prompt_embeds=embedding, num_inference_steps=50, width=width, generator=generator)
    image = out2.images[0]

    # Convert spectrogram image back to audio
    params = SpectrogramParams()
    converter = SpectrogramImageConverter(params=params)

    segment = converter.audio_from_spectrogram_image(image, apply_filters=True, )

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    name = None
    if name is None:
        audio_filename = f"generated_music_{torch.randint(0, int(1e6), (1,)).item()}.wav"
    else:
        audio_filename = f"{name}.wav"
    audio_path = os.path.join(output_dir, audio_filename)
    segment.export(audio_path, format="wav")
    # breakpoint()

    print("done")

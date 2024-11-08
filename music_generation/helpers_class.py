import os
from typing import Optional, Union

import transformers
import torch
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch import nn
from diffusers import (DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler, PNDMScheduler,
                       LMSDiscreteScheduler)
from transformers import CLIPTokenizer, CLIPTextModel, AutoTokenizer, CLIPFeatureExtractor

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
        self.tokenizer = AutoTokenizer.from_pretrained(MUSICGEN_MODEL_ID)
        self.model = transformers.MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL_ID)

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        inputs = self.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.model.get_encoder()(**inputs).last_hidden_state

    def token_to_embed(self, tokens):
        with torch.no_grad():
            return self.model.get_encoder()(**tokens).last_hidden_state


if __name__ == "__main__":
    dh = RiffusionPipeline.from_pretrained(RIFFUSION_MODEL_ID)

    a = dh.text_to_embed("hello world how is it going", 10)
    b = dh.text_to_embeddings_before_clip("hello world how is it going", 10)
    breakpoint()

    print("done")

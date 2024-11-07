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


class EasyDiffuse(DiffusionPipeline):
    def __init__(self):
        super(EasyDiffuse, self).__init__()

    def text_to_embed(self, text):
        raise NotImplementedError

    def token_to_embed(self, tokens):
        raise NotImplementedError


class RiffusionPipeline(EasyDiffuse):
    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet2DConditionModel,
            scheduler: DDIMScheduler | PNDMScheduler | LMSDiscreteScheduler,
            safety_checker: StableDiffusionSafetyChecker,
            feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        inputs = self.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embedded_text = self.text_encoder(inputs.input_ids.to(self.device))
        return embedded_text

    def token_to_embed(self, input_ids):
        with torch.no_grad():
            return self.text_encoder(input_ids.to(self.device))

    def text_to_embeddings_before_clip(self, text, max_length=None):
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        inputs = self.tokenizer(text, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            return self.text_encoder.get_input_embeddings()(inputs.input_ids.to(self.device))


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

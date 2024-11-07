import transformers
import torch
from torch import nn
from diffusers import DiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel

RIFFUSION_MODEL_ID = "riffusion/riffusion-model-v1"
MUSICGEN_MODEL_ID = "facebook/musicgen-small"


class EasyDiffuse(nn.Module):
    def __init__(self):
        super(EasyDiffuse, self).__init__()

    def text_to_embed(self, text):
        raise NotImplementedError

    def token_to_embed(self, tokens):
        raise NotImplementedError


class RiffusionHelpers(EasyDiffuse):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained(RIFFUSION_MODEL_ID)
        self.text_encoder = CLIPTextModel.from_pretrained(RIFFUSION_MODEL_ID)
        self.model = DiffusionPipeline.from_pretrained(RIFFUSION_MODEL_ID)

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
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MUSICGEN_MODEL_ID)
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
    dh = RiffusionHelpers()
    dh.text_to_embed_musicgen("hello")
    dh.text_to_embed_riffusion("hello")
    dh.token_to_embed_riffusion("hello")
    breakpoint()
    print("done")

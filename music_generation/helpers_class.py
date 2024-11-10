import math
import os
import scipy
import torch
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration, T5EncoderModel

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from diffusers.utils.testing_utils import enable_full_determinism

RIFFUSION_MODEL_ID = "riffusion/riffusion-model-v1"
MUSICGEN_MODEL_ID = "facebook/musicgen-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST = "musicgen"  # "riffusion"


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
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True,
                                      return_tensors="pt")
        with torch.no_grad():
            embedded_text = self.model.text_encoder(inputs.input_ids.to(self.model.device))
        return embedded_text.last_hidden_state

    def token_embedding_to_embed(self, token_embedding):
        with torch.no_grad():
            encoded = self.model.text_encoder.text_model.encoder(
                token_embedding.to(self.model.device)).last_hidden_state
            return self.model.text_encoder.text_model.final_layer_norm(encoded)

    def text_to_embeddings_before_clip(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True,
                                      return_tensors="pt")
        with torch.no_grad():
            return self.model.text_encoder.text_model.embeddings(inputs.input_ids.to(self.model.device))

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class MusicGenPipeline(EasyDiffuse):
    def __init__(self):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL_ID)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL_ID)

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(text, padding="max_length", max_length=max_length, truncation=True,
                                      return_tensors="pt")
        with torch.no_grad():
            return self.model.get_encoder()(**inputs).last_hidden_state

    def token_embedding_to_embed(self, **token_embedding):
        with torch.no_grad():
            return self.model.text_encoder.encoder(**token_embedding)

    def text_to_embeddings_before_encoder(self, input_ids, **kwargs):
        with torch.no_grad():
            return self.model.get_input_embeddings()(input_ids)


if __name__ == "__main__":

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    name = "musicgen_out" if TEST == "musicgen" else "riffusion_out"
    if name is None:
        audio_filename = f"generated_music_{torch.randint(0, int(1e6), (1,)).item()}.wav"
    else:
        audio_filename = f"{name + '_' + str(len(os.listdir(output_dir)))}.wav"

    audio_path = os.path.join(output_dir, audio_filename)

    if TEST == "riffusion":
        enable_full_determinism()
        generator = torch.Generator(device=DEVICE)
        width = math.ceil(3 * (512 / 5))
        width = width + (8 - width % 8) if width % 8 != 0 else width
        generator.manual_seed(0)
        riffusion_pipe = EasyRiffPipeline()
        embedding = riffusion_pipe.text_to_embed(
            "Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive.",
            30)
        output = riffusion_pipe(prompt_embeds=embedding, num_inference_steps=50, width=width, generator=generator)
        image = output.images[0]
        # Test embeddings pre-clip
        # embedding_pre = riffusion_pipe.text_to_embeddings_before_clip("Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive.", 30)
        # embedding = riffusion_pipe.token_embedding_to_embed(embedding_pre)
        # out2 = riffusion_pipe(prompt_embeds=embedding, num_inference_steps=50, width=width, generator=generator)
        # image = out2.images[0]

        # Convert spectrogram image back to audio
        params = SpectrogramParams()
        converter = SpectrogramImageConverter(params=params)
        segment = converter.audio_from_spectrogram_image(image, apply_filters=True, )
        segment.export(audio_path, format="wav")
    elif TEST == "musicgen":
        musicgen_pipe = MusicGenPipeline()

        inputs_gen = musicgen_pipe.processor(
            text=[
                "Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive."],
            padding=True,
            return_tensors="pt",
        )
        x = musicgen_pipe.text_to_embeddings_before_encoder(**inputs_gen)
        x = {"inputs_embeds": x,
             "attention_mask": inputs_gen["attention_mask"]}  # Not sure to let attention mask as is or to remove it

        audio_values = musicgen_pipe.model.generate(inputs_embeds=x, max_new_tokens=256)
        sampling_rate = musicgen_pipe.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(audio_path, rate=sampling_rate, data=audio_values[0, 0].numpy())
    else:
        raise ValueError("TEST must be either 'riffusion' or 'musicgen'")

    print("done")

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
TEST = "riffusion"  # "riffusion"


def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


class EasyDiffuse:
    def __init__(
        self, input_type="text", output_dir="generated_audio", exp_name="test"
    ):
        super().__init__()
        self.input_type = input_type
        self.output_dir = output_dir
        self.exp_name = exp_name

    def text_to_embed(self, inputs):
        raise NotImplementedError

    def token_embedding_to_embed(self, token_embedding):
        raise NotImplementedError

    def text_to_embeddings_before_encoder(self, inputs):
        raise NotImplementedError

    def generate_music(self, embeddings, **kwargs):
        raise NotImplementedError

    def transform_inputs(self, inputs):
        raise NotImplementedError

    def generate_path(self):
        return os.path.join(
            self.output_dir,
            f"{self.exp_name + '_' + str(len(os.listdir(self.output_dir)))}.wav",
        )


class EasyRiffPipeline(EasyDiffuse):
    def __init__(
        self,
        input_type="text",
        output_dir="generated_audio",
        exp_name="test",
        generator=None,
        inference_steps=50,
    ):
        super().__init__(input_type, output_dir=output_dir, exp_name=exp_name)
        self.width = math.ceil(3 * (512 / 5))
        self.width = (
            self.width + (8 - self.width % 8) if self.width % 8 != 0 else self.width
        )
        self.generator = generator
        self.inference_steps = inference_steps
        self.model = DiffusionPipeline.from_pretrained(RIFFUSION_MODEL_ID).to(DEVICE)

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embedded_text = self.model.text_encoder(
                inputs.input_ids.to(self.model.device)
            )
        return embedded_text.last_hidden_state

    def token_embedding_to_embed(self, token_embedding):
        with torch.no_grad():
            encoded = self.model.text_encoder.text_model.encoder(
                token_embedding.to(self.model.device)
            ).last_hidden_state
            return self.model.text_encoder.text_model.final_layer_norm(encoded)

    def text_to_embeddings_before_encoder(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            return self.model.text_encoder.text_model.embeddings(
                inputs.input_ids.to(self.model.device)
            )

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def transform_inputs(self, inputs):
        if self.input_type == "text":
            return self.text_to_embed(inputs)
        elif self.input_type == "token_embeddings":
            return self.token_embedding_to_embed(inputs)
        elif self.input_type == "embeddings":
            return inputs
        else:
            raise ValueError(
                "input_type must be either 'text', 'token_embedding', or 'embeddings'"
            )

    def generate_music(self, inputs, **kwargs):
        embeddings = self.transform_inputs(inputs)
        output = self.model(
            prompt_embeds=embeddings,
            generator=self.generator,
            safety_checker=dummy_safety_checker,
            num_inference_steps=self.inference_steps,
            **kwargs,
        )
        audio_path = self.generate_path()
        image = output.images[0]
        params = SpectrogramParams()
        converter = SpectrogramImageConverter(params=params)
        segment = converter.audio_from_spectrogram_image(image, apply_filters=True)
        segment.export(audio_path, format="wav")
        return audio_path


class MusicGenPipeline(EasyDiffuse):
    def __init__(
        self, input_type="text", output_dir="generated_audio", exp_name="test"
    ):
        super().__init__(input_type, output_dir=output_dir, exp_name=exp_name)
        self.processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL_ID)
        self.model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL_ID)

    def text_to_embed(self, text, max_length=None):
        if max_length is None:
            max_length = self.processor.tokenizer.model_max_length
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            return self.model.get_encoder()(**inputs).last_hidden_state

    def token_embedding_to_embed(self, token_embedding):
        with torch.no_grad():
            return self.model.text_encoder.encoder(**token_embedding)

    def text_to_embeddings_before_encoder(self, inputs, max_length=None):
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )

        with torch.no_grad():
            return self.model.get_input_embeddings()(inputs["input_ids"])

    def transform_inputs(self, inputs):
        if self.input_type == "text":
            return self.text_to_embed(inputs)
        elif self.input_type == "token_embeddings":
            inputs = {"inputs_embeds": inputs}
            return self.token_embedding_to_embed(inputs).last_hidden_state
        elif self.input_type == "embeddings":
            return inputs
        else:
            raise ValueError(
                "input_type must be either 'text', 'token_embedding', or 'embeddings'"
            )

    def generate_music(self, inputs, **kwargs):
        embeddings = self.transform_inputs(inputs)
        audio_path = self.generate_path()
        audio_values = self.model.generate(inputs_embeds=embeddings, **kwargs)
        sampling_rate = musicgen_pipe.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(
            audio_path, rate=sampling_rate, data=audio_values[0, 0].numpy()
        )
        return audio_path


if __name__ == "__main__":

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    name = "musicgen_out" if TEST == "musicgen" else "riffusion_out"
    enable_full_determinism()
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(0)
    text = "Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb. Imagine a neon-lit night drive."

    if TEST == "riffusion":
        width = math.ceil(3 * (512 / 5))
        width = width + (8 - width % 8) if width % 8 != 0 else width
        # ---------------- Test with text directly ----------------
        riffusion_pipe = EasyRiffPipeline(
            input_type="text",
            output_dir=output_dir,
            exp_name="riffusion_text",
            generator=generator,
            inference_steps=10,
        )

        riffusion_pipe.generate_music(text)
        # ---------------- Test with embeds directly ----------------
        riffusion_pipe = EasyRiffPipeline(
            input_type="embeddings",
            output_dir=output_dir,
            exp_name="riffusion_embeddings",
            generator=generator,
            inference_steps=10,
        )
        embedding = riffusion_pipe.text_to_embed(text, 30)
        riffusion_pipe.generate_music(embedding)

        # ---------------- Test with pre Clip ----------------
        riffusion_pipe = EasyRiffPipeline(
            input_type="token_embeddings",
            output_dir=output_dir,
            exp_name="riffusion_preclip",
            generator=generator,
            inference_steps=10,
        )
        embedding_pre = riffusion_pipe.text_to_embeddings_before_encoder(text, 30)
        riffusion_pipe.generate_music(embedding)

    elif TEST == "musicgen":
        # ---------------- Test with text directly ----------------
        musicgen_pipe = MusicGenPipeline("text", output_dir, "musicgen_text")
        path = musicgen_pipe.generate_music(text, max_new_tokens=256)

        # ---------------- Test with embeds directly ----------------
        inputs_gen = musicgen_pipe.text_to_embed(text, max_length=256)
        musicgen_pipe = MusicGenPipeline("embeddings", output_dir, "embeddings")
        path = musicgen_pipe.generate_music(inputs_gen, max_new_tokens=256)
        # ---------------- Test with pre T5 ----------------
        inputs_gen = musicgen_pipe.text_to_embeddings_before_encoder(
            text, max_length=256
        )

        musicgen_pipe = MusicGenPipeline(
            "token_embeddings", output_dir, "token_embeddings"
        )
        path = musicgen_pipe.generate_music(inputs_gen, max_new_tokens=256)

    else:
        raise ValueError("TEST must be either 'riffusion' or 'musicgen'")

    print("done")

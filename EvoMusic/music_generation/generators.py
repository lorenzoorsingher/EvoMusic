import math
import os
import scipy
import torch
from typing import Union
from diffusers import DiffusionPipeline
from transformers import AutoProcessor, MusicgenForConditionalGeneration, set_seed

from diffusers.utils.testing_utils import enable_full_determinism

import EvoMusic.configuration as c
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams

def dummy_safety_checker(images, **kwargs):
    return images, [False] * len(images)


class MusicGenerator:
    """
    Base class for music generation, contains the main methods to be implemented by the subclasses.
    """

    def __init__(self, music_generator: c.MusicGeneratorConfig):
        super().__init__()
        self.config = music_generator

    def text_to_embed(self, text: str, max_length: int = None):
        """
        Takes text and returns the embeddings for the model

            Args:
                text (str): text input
                max_length (int, optional): max length of the generated sequence. Defaults to None.
        """
        raise NotImplementedError

    def token_embedding_to_embed(self, token_embedding: torch.Tensor):
        """
        Takes token embeddings from the tokenizer and returns the embeddings for the model
            Args:
                token_embedding (torch.Tensor): token embeddings from the tokenizer
            Returns:
                torch.Tensor: embeddings for the model, usually output of the whole encoder part of the model
        """
        raise NotImplementedError

    def text_to_embeddings_before_encoder(self, text: str, max_length: int = None):
        """
        Takes text and returns the embeddings before the encoder of the model.
        These are usually the token_embeddings from the tokenizer and the first embedding layer of the model.
            Args:
                text (str): text input
                max_length (int, optional): max length of the generated sequence. Defaults to None.
            Returns:
                torch.Tensor: embeddings before the encoder
        """
        raise NotImplementedError

    def get_embedding_size(self):
        """
        Returns the size of the embeddings for the specific model
            Returns:
                int: size of the embeddings
        """
        raise NotImplementedError

    def generate_music(
        self, input: Union[torch.Tensor, str], duration: int, name: str = None, **kwargs
    ):
        """
        Generates music from the input
            Args:
                input (str | torch.Tensor): input for the model
                duration (int): duration in seconds of the generated audio
                name (str, optional): name of the generated audio
            Returns:
                str: system path to the generated audio
        """
        raise NotImplementedError

    def prepare_inputs(self, text: str, max_length: int = None):
        """
        Prepares the inputs for the 'transform_inputs' function
        It uses the tokenizer of the model to tokenize the text and return the inputs for the user, so that calculation
        can be done on it.
            Args:
                text (str): text input
                max_length (int, optional): max length of the generated sequence. Defaults to None.
            Returns:
                dict: inputs for the model
        """
        raise NotImplementedError

    def transform_inputs(self, inputs: Union[str, torch.Tensor]):
        """
        Transforms the inputs to the embeddings that can be used by the model
            Args:
                inputs (str | torch.Tensor): inputs to be transformed
            Returns:
                torch.Tensor: embeddings for the model
        """
        raise NotImplementedError

    def preprocess_text(self, text: list[str], max_length: int = None):
        """
        Converts the text to the format that the model can understand
            Args:
                text (list[str]): text input
                max_length (int, optional): max length of the generated sequence. Defaults to None.
            Returns:
                str|torch.Tensor: processed text
        """
        if self.config.input_type == "text":
            return text
        elif self.config.input_type == "token_embeddings":
            return [self.text_to_embeddings_before_encoder(t, max_length).squeeze(0) for t in text]
        elif self.config.input_type == "embeddings":
            return [self.text_to_embed(t, max_length).squeeze(0) for t in text]
        
    def generate_path(self, name=None):
        """
        Generates the path for the output audio, if name is None, it will use jut the default experiment name
        The number is given based on the number of files in the output directory.
            Args:
                name (str, optional): name of the file. Defaults to None.
            Returns:
                str: system path to the generated audio
        """
        # check if the output directory exists
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        
        base_name = self.config.name if name is None else name + "_" + self.config.name
        return os.path.join(
            self.config.output_dir,
            f"{base_name + '_' + str(len(os.listdir(self.config.output_dir)))}.wav",
        )


class EasyRiffPipeline(MusicGenerator):
    def __init__(self, riffusion_config: c.EasyRiffusionConfig):
        super().__init__(riffusion_config)
        self.model = DiffusionPipeline.from_pretrained(self.config.model).to(
            self.config.device
        )
        self.model.safety_checker = dummy_safety_checker

    def text_to_embed(self, text, max_length=None):
        inputs = self.prepare_inputs(text, max_length)
        with torch.no_grad():
            embedded_text = self.model.text_encoder(
                inputs.input_ids.to(self.model.device)
            )
        return embedded_text.last_hidden_state.to(self.model.device)

    def token_embedding_to_embed(self, token_embedding):
        with torch.no_grad():
            encoded = self.model.text_encoder.text_model.encoder(
                token_embedding.to(self.model.device)
            ).last_hidden_state
            return self.model.text_encoder.text_model.final_layer_norm(encoded)

    def text_to_embeddings_before_encoder(self, text, max_length=None):
        inputs = self.prepare_inputs(text, max_length)
        with torch.no_grad():
            return self.model.text_encoder.text_model.embeddings(
                inputs.input_ids.to(self.model.device)
            )

    def get_embedding_size(self):
        return self.model.text_encoder.text_model.config.hidden_size

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def prepare_inputs(self, text, max_length=None):
        if max_length is None:
            max_length = self.model.tokenizer.model_max_length
        inputs = self.model.tokenizer(
            text,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def transform_inputs(self, inputs):
        emb_size = self.get_embedding_size()
        if isinstance(inputs, torch.Tensor):
            i = inputs.view(1, -1, emb_size).to(device=self.model.device)
        else:
            i = inputs

        if self.config.input_type == "text":
            return self.text_to_embed(i)
        elif self.config.input_type == "token_embeddings":
            return self.token_embedding_to_embed(i)
        elif self.config.input_type == "embeddings":
            return i
        else:
            raise ValueError(
                "input_type must be either 'text', 'token_embedding', or 'embeddings'"
            )

    def generate_music(self, input, duration=5, name=None, **kwargs):
        embeddings = self.transform_inputs(input)
        width = math.ceil(duration * (512 / 5))
        generator = torch.Generator(device=self.model.device)
        generator.manual_seed(0)
        output = self.model(
            prompt_embeds=embeddings,
            generator=generator,
            num_inference_steps=self.config.inference_steps,
            width=width,
            **kwargs,
        )
        audio_path = self.generate_path(name)
        image = output.images[0]
        params = SpectrogramParams()
        converter = SpectrogramImageConverter(params=params)
        segment = converter.audio_from_spectrogram_image(image, apply_filters=True)
        segment.export(audio_path, format="wav")
        return audio_path


class MusicGenPipeline(MusicGenerator):
    def __init__(self, musicgen_config: c.MusicGeneratorConfig):
        super().__init__(musicgen_config)
        self.processor = AutoProcessor.from_pretrained(self.config.model)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            self.config.model
        ).to(self.config.device)
        self.model.eval()

    def text_to_embed(self, text, max_length=None):
        inputs = self.prepare_inputs(text, max_length)

        with torch.no_grad():
            # TODO: Not Working right now, need to fix
            return self.model._prepare_text_encoder_kwargs_for_generation(
                inputs["input_ids"]
            )

    def text_to_embeddings_before_encoder(self, text, max_length=None):
        inputs = self.prepare_inputs(text, max_length)

        with torch.no_grad():
            return self.model.get_input_embeddings()(inputs["input_ids"])

    def get_embedding_size(self):
        return self.model.text_encoder.config.hidden_size

    def prepare_inputs(self, text, max_length=None):
        if max_length is None:
            max_length = self.processor.tokenizer.model_max_length
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        for key in inputs:
            if isinstance(inputs[key], torch.Tensor):
                inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def transform_inputs(self, inputs):
        emb_size = self.get_embedding_size()
        if isinstance(inputs, torch.Tensor):
            i = inputs.view(1, -1, emb_size).to(device=self.model.device)
        else:
            i = inputs
        if self.config.input_type == "text":
            outputs = self.processor(text=[i], return_tensors="pt")
            for key in outputs:
                if isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].to(self.model.device)
            return outputs
        elif self.config.input_type == "token_embeddings":
            return {"inputs_embeds": i}
        elif self.config.input_type == "embeddings":
            i.last_hidden_state = torch.concatenate(
                [i.last_hidden_state, torch.zeros_like(i.last_hidden_state)],
                dim=0,
            )
            return {"encoder_outputs": i}

    def generate_music(self, input, duration=5, name=None, **kwargs):
        embeddings = self.transform_inputs(input)
        audio_path = self.generate_path(name)
        set_seed(0)
        kwargs["max_new_tokens"] = int(duration / 5 * 256)
        audio_values = self.model.generate(
            **embeddings, **kwargs, do_sample=True, top_k=0
        )
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(
            audio_path, rate=sampling_rate, data=audio_values[0, 0].cpu().numpy()
        )
        return audio_path


if __name__ == "__main__":
    config = c.load_yaml_config("example_conf/test_music_generation_config.yaml")

    TEST = "musicgen"  # "riffusion"s

    output_dir = "generated_audio"
    os.makedirs(output_dir, exist_ok=True)
    name = "musicgen_out" if TEST == "musicgen" else "riffusion_out"
    enable_full_determinism()

    txt = "Create a retro 80s synthwave track with nostalgic synthesizers, a steady electronic beat, and atmospheric reverb."

    if TEST == "riffusion":
        # ---------------- Test with text directly ----------------
        cfg = config.riffusion_pipeline

        riffusion_pipe = EasyRiffPipeline(cfg)
        print(riffusion_pipe.model.text_encoder.text_model.config)
        # print(riffusion_pipe.model)
        print(riffusion_pipe.get_embedding_size())
        riffusion_pipe.generate_music(txt)
        # ---------------- Test with embeds directly ----------------
        cfg.exp_name = "riffusion_embeds"
        cfg.input_type = "embeddings"

        riffusion_pipe = EasyRiffPipeline(cfg)
        embedding = riffusion_pipe.text_to_embed(txt)
        riffusion_pipe.generate_music(embedding)

        # ---------------- Test with pre Clip ----------------
        cfg.exp_name = "riffusion_clip"
        cfg.input_type = "token_embeddings"

        riffusion_pipe = EasyRiffPipeline(cfg)
        embedding_pre = riffusion_pipe.text_to_embeddings_before_encoder(txt)
        riffusion_pipe.generate_music(embedding)

    elif TEST == "musicgen":
        cfg = config.music_generator
        # ---------------- Test with text directly ----------------
        musicgen_pipe = MusicGenPipeline(cfg)
        path = musicgen_pipe.generate_music(txt, max_new_tokens=256)
        print()

        # ---------------- Test with pre T5 ----------------
        inputs_gen = musicgen_pipe.text_to_embeddings_before_encoder(txt)
        cfg.exp_name = "musicgen_pre"
        cfg.input_type = "token_embeddings"
        musicgen_pipe = MusicGenPipeline(cfg)
        path = musicgen_pipe.generate_music(inputs_gen, max_new_tokens=256)

        # ---------------- Test with embeds ----------------
        # cfg.exp_name = "musicgen_embeds"
        # cfg.input_type = "embeddings"
        # musicgen_pipe = MusicGenPipeline(cfg)
        # inputs_gen = musicgen_pipe.text_to_embed(txt, max_length=256)
        # path = musicgen_pipe.generate_music(inputs_gen, max_new_tokens=256)

    else:
        raise ValueError("TEST must be either 'riffusion' or 'musicgen'")

    print("done")

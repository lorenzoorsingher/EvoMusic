import os
import torch
import torchaudio

from transformers import AutoModel, Wav2Vec2FeatureExtractor

if __name__ == "__main__":
    import sys

    sys.path.append("./")
    sys.path.append("../")

from configuration import ProjectConfig, FitnessConfig

from usrembeds.models.model import AlignerV2


class MusicScorer:
    """
    Class to compute the fitness of a song based either on a user or a target song
    """

    def __init__(self, fitness_config: FitnessConfig):
        """
        Args:
            fitness_config (FitnessConfig): The fitness configuration
        """
        self.config = fitness_config

        assert self.config.mode in ["user", "music", "dynamic"], "Invalid fitness mode"

        self.music_embedder_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        ).to(self.config.device)
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-95M", trust_remote_code=True
        )
        self.resample_rate = self.processor.sampling_rate

        state_dict, conf, _ = AlignerV2.load_model(
            "usrembeds/checkpoints/AlignerV2_best_model.pt", self.config.device
        )
        model_conf = {
            "emb_size": conf["emb_size"],
            "prj_size": conf["prj_size"],
            "prj_type": conf["prj"],
            "aggragation": conf["aggr"],
            "n_users": conf["nusers"],
            "lt": conf["learnable_temp"],
        }
        self.model = AlignerV2(**model_conf).to(self.config.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        if self.config.mode == "music":
            # check if the target music exists
            assert os.path.exists(
                self.config.target_music
            ), "Target music does not exist"
            self.target_music_emb = self.embed_audios([self.config.target_music]).view(
                1, -1
            )

    def embed_audios(self, audio_paths: list[str]) -> torch.Tensor:
        """
        Compute the embedding of an audio file using the EncodecModel.
        Ensures that the audio is stereo (2 channels) before processing.

        Args:
            audio_paths (list[str]): The paths to the audio files to embed

        Returns:
            torch.Tensor: The embedding of the audio file
        """
        embeddings = []
        for audio_path in audio_paths:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample audio if necessary
            if sample_rate != self.resample_rate:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.resample_rate
                )
                waveform = resampler(waveform)
                sample_rate = self.resample_rate

            # make audio mono if stereo
            if waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0)
            else:
                waveform = waveform.squeeze(0)

            with torch.no_grad():
                inputs = self.processor(
                    waveform, sampling_rate=self.resample_rate, return_tensors="pt"
                )
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
                output = self.music_embedder_model(**inputs, output_hidden_states=True)
                embedding = torch.stack(output.hidden_states).mean(dim=2).view(13, -1)

            embeddings.append(embedding)

        return torch.stack(embeddings)

    def get_user_likeness(self, audio_paths: list[str], user_idx: int = 0):
        """
        Compute the likeness of a user to an audio file

        Args:
            audio_paths (list[str]): The paths to the audio files to compare
            user_idx (int): The index of the user to compare to

        Returns:
            torch.Tensor: The likeness of the user to each audio files
        """
        music_embs = self.embed_audios(audio_paths)
        music_embs = music_embs.unsqueeze(0)  # shape (1, n, emb_size)

        # if user_idx is not a tensor, convert it to a tensor
        if not torch.is_tensor(user_idx):
            user_idx = torch.tensor(user_idx).to(self.config.device).unsqueeze(0)

        with torch.no_grad():
            usr, embs, temp = self.model(user_idx, music_embs)
            embs = embs.squeeze(0)  # shape (n, prj_size)
            return torch.cosine_similarity(usr, embs, dim=1)

    def compute_fitness(self, audio_paths: list[str]):
        """
        Compute the fitness of the audio files

        Args:
            audio_paths (list[str]): The paths to the audio files to compare

        Returns:
            torch.Tensor: The fitness of the audio files
        """
        if self.config.mode == "user":
            return self.get_user_likeness(audio_paths, self.config.target_user)

        elif self.config.mode == "music":
            music_embs = self.embed_audios(audio_paths).view(len(audio_paths), -1)
            with torch.no_grad():
                return torch.cosine_similarity(music_embs, self.target_music_emb, dim=1)

        elif self.config.mode == "dynamic":
            raise NotImplementedError("Dynamic mode is not yet implemented")


if __name__ == "__main__":
    fitness_config = FitnessConfig()

    print("-----------------------------------")
    print("Testing MusicScorer in mode music")
    print("-----------------------------------")
    fitness_config.mode = "music"
    fitness_config.target_music = "./generated_audio/default_10.wav"
    music_scorer = MusicScorer(fitness_config)
    print(music_scorer(["./generated_audio/best_music.wav"]))

    print("-----------------------------------")
    print("Testing MusicScorer in mode user")
    print("-----------------------------------")
    fitness_config.mode = "user"
    fitness_config.target_user = 0
    music_scorer = MusicScorer(fitness_config)
    print(music_scorer(["./generated_audio/best_music.wav"]))

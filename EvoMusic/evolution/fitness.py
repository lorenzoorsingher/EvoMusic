import os
import torch
import torchaudio

from transformers import AutoModel, Wav2Vec2FeatureExtractor

from EvoMusic.configuration import ProjectConfig, FitnessConfig

from EvoMusic.user_embs.model import AlignerV2

import librosa
import numpy as np
from scipy.signal import butter, filtfilt

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

    def measure_audio_artifacts(self, audio_paths: list[str]) -> torch.Tensor:
        """
        Detect and measure audio artifacts in generated music with more granular analysis,
        focusing on clipping, DC offset, high-frequency noise, and abrupt discontinuities.
        
        Returns a penalty score in [0, 1] for each audio file.
        """
        penalties = []

        # Hyperparameters and weights (tunable)
        FRAME_SIZE = 1024
        HOP_SIZE = 512

        w_clipping = 0.2
        w_dc_offset = 0.1
        w_flux = 0.2
        w_hf_noise = 0.2
        w_silence = 0.2

        for audio_path in audio_paths:
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            # -----------------------------------------------------------
            # 1. Frame-based Analysis
            # -----------------------------------------------------------
            # We’ll split the audio into frames for more detailed analysis
            # so that abrupt events are captured.
            frames = librosa.util.frame(
                y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE
            ).T  # shape: (#frames, FRAME_SIZE)

            # Safeguard: If the audio is shorter than one frame, pad or skip
            if frames.shape[0] < 1:
                penalties.append(0.5)  # Arbitrary penalty for very short audio
                continue

            # -----------------------------------------------------------
            # 2. Clipping Detection
            # -----------------------------------------------------------
            # Count how many samples are near +1.0 or -1.0 in each frame
            clipping_threshold = 0.99
            clipping_counts = np.sum(np.abs(frames) > clipping_threshold, axis=1)
            # Turn into ratio of clipped samples per frame
            clipping_ratio_per_frame = clipping_counts / FRAME_SIZE
            # Take average clipping ratio across frames
            avg_clipping_ratio = np.mean(clipping_ratio_per_frame)
            # Convert to penalty, capping at 1.0
            clipping_penalty = np.clip(avg_clipping_ratio * 5.0, 0, 1)

            # -----------------------------------------------------------
            # 3. DC Offset
            # -----------------------------------------------------------
            # DC offset per frame is the mean of each frame
            dc_offsets = np.mean(frames, axis=1)
            avg_dc_offset = np.mean(np.abs(dc_offsets))
            # Convert to penalty (tunable)
            # The scale factor below is somewhat arbitrary, you can experiment
            dc_penalty = np.clip(avg_dc_offset * 10.0, 0, 1)

            # -----------------------------------------------------------
            # 4. High-Frequency Content / Harsh Noise
            # -----------------------------------------------------------
            # Compute a short-time Fourier transform or a simple spectrum for each frame.
            # We'll measure the fraction of energy above a threshold frequency.
            # For instance, above 8 kHz could be considered "high" for certain musical contexts.
            # If sr < 16000, adjust accordingly.
            stft = librosa.stft(y, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, window='hann')
            magnitude = np.abs(stft)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=FRAME_SIZE)
            
            # define a high-frequency threshold, e.g., 8 kHz
            hf_threshold = 8000  
            hf_indices = freqs >= hf_threshold

            # sum energy in high freq bands
            hf_energy = np.sum(magnitude[hf_indices, :], axis=0)
            total_energy = np.sum(magnitude, axis=0) + 1e-8  # avoid /0
            hf_ratio = hf_energy / total_energy
            # average HF ratio across frames
            avg_hf_ratio = np.mean(hf_ratio)
            # turn it into a penalty if it’s abnormally high
            hf_noise_penalty = np.clip(avg_hf_ratio * 3.0, 0, 1)

            # -----------------------------------------------------------
            # 5. Spectral Flux for Discontinuity
            # -----------------------------------------------------------
            # We still use librosa's onset_strength or compute flux ourselves:
            flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_SIZE)
            # measure variation in flux
            flux_std = np.std(flux)
            flux_mean = np.mean(flux) + 1e-8
            flux_penalty = np.clip(flux_std / flux_mean, 0, 1)

            # -----------------------------------------------------------
            # 6. Silence or Prolonged Near-Silence
            # -----------------------------------------------------------
            # We'll compute RMS in short frames, check the ratio of near-silence frames
            rms = librosa.feature.rms(y=y, frame_length=FRAME_SIZE, hop_length=HOP_SIZE)[0]
            # threshold for near-silence
            silence_threshold = np.percentile(rms, 10)  # e.g. 10th percentile
            silent_frames = rms < silence_threshold
            silence_ratio = np.mean(silent_frames)
            # turn into penalty, with some cap
            silence_penalty = np.clip(silence_ratio * 2.0, 0, 1)

            # -----------------------------------------------------------
            # 7. Combine Penalties
            # -----------------------------------------------------------
            # Weighted sum (one of many ways to combine metrics)
            total_penalty = (
                w_clipping * clipping_penalty
                + w_dc_offset * dc_penalty
                + w_hf_noise * hf_noise_penalty
                + w_flux * flux_penalty
                + w_silence * silence_penalty
                # if you want to incorporate other smaller metrics,
                # you can place them here under w_misc
            )

            # Bound the total penalty in [0, 1]
            total_penalty = np.clip(total_penalty, 0.0, 1.0)

            penalties.append(total_penalty)

        return torch.tensor(penalties, dtype=torch.float32)

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
        # 1) Compute base fitness (likeness to user, music, etc.).
        if self.config.mode == "user":
            base = self.get_user_likeness(audio_paths, self.config.target_user)

        elif self.config.mode == "music":
            music_embs = self.embed_audios(audio_paths).view(len(audio_paths), -1)
            base = torch.cosine_similarity(music_embs, self.target_music_emb, dim=1)

        elif self.config.mode == "dynamic":
            raise NotImplementedError("Dynamic mode is not yet implemented")
        
        # 2) Compute artifact penalties (graininess, clipping, etc.).
        penalties = self.measure_audio_artifacts(audio_paths).to(base.device)
        
        
        # 3) Combine base fitness with penalty.
        noise_weight = self.config.noise_weight  # Or however you've configured it.
        final_fitness = base - noise_weight * penalties

        return final_fitness


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

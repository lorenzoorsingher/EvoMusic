import torch
from torch.utils.data import DataLoader

from EvoMusic.usrapprox.utils.memory import MemoryBuffer


class User:
    def __init__(
        self,
        user_id: int,
        # user_in_model_id: int,
    ):
        self._uuid = user_id
        self._user_id = None
        self._memory: MemoryBuffer = None
        self._minibatch: bool = None

        self._train_dataloader = None
        self._test_dataloader = None

    def set_minibatch(self, minibatch: bool):
        if self._minibatch is not None:
            raise ValueError("Minibatch already set.")
        self._minibatch = minibatch

    def set_memory_size(self, memory_length: int):
        if self._memory is not None:
            raise ValueError("Memory already set.")
        self._memory = MemoryBuffer(memory_length)

    def set_memory_device(self, device):
        self._memory.to(device)

    def set_user_id(self, user_id: int):
        if self._user_id is not None:
            raise ValueError("User ID already set.")
        self._user_id = user_id

    def add_to_memory(self, batch: torch.Tensor, feedback: torch.Tensor):
        self._memory.add_to_memory(batch, feedback)

    def empty_memory(self):
        self._memory.empty_memory()

    def set_train_dataloader(
        self, train_dataloader: DataLoader
    ):
        if self._train_dataloader is not None:
            raise ValueError("Train Dataloader already set.")

        self._train_dataloader = train_dataloader

    def set_test_dataloader(
        self, test_dataloader: DataLoader
    ):
        if self._test_dataloader is not None:
            raise ValueError("Test Dataloader already set.")

        self._test_dataloader = test_dataloader

    @property
    def uuid(self) -> int:
        return self._uuid

    @property
    def user_id(self) -> int:
        """
        get the user delta.
        """
        return self._user_id

    @property
    def memory(self) -> torch.Tensor:
        """
        get the memory.
        """
        return self._memory.memory

    @property
    def user_id(self) -> int:
        """
        get the user id.
        """
        return self._user_id

    @property
    def train_dataloader(self) -> DataLoader:
        """
        get the train dataloader.
        """
        if self._train_dataloader is None:
            raise ValueError("Train Dataloader not set.")

        return self._train_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        """
        get the test dataloader.
        """

        return self._test_dataloader
    
    @property
    def minibatch(self) -> bool:
        """
        get the minibatch.
        """
        return self._minibatch


class SynthUser(User):
    def __init__(
        self,
        user_id: int,
    ):
        super().__init__(user_id)
        self._model_reference_id = None

    def set_user_reference(self, reference_id: int):
        if self._model_reference_id is not None:
            raise ValueError("User delta already set.")
        self._model_reference_id = reference_id

    @property
    def model_reference_id(self) -> int:
        return self._model_reference_id


class RealUser(User):
    def __init__(
        self,
        user_id: int,
    ):
        super().__init__(user_id)
        self._playlist = None
        self._feedbacks = []
        self._prev_playlist = None
        self._model_reference_id = "real_user"
        
    @property
    def model_reference_id(self) -> str:
        return self._model_reference_id
        
    def set_playlist(self, playlist: list[str]):
        """Set the playlist of the user.

        Args:
            playlist (list[str]): The playlist of the user as list of file paths.
        """
        self._playlist = playlist

    def evaluate_playlist(self) -> torch.Tensor:
        """
        Launch a GUI that plays each song from the playlist and collects user feedback.
        For each song, the user is shown the song's file path, the song is played,
        and the user clicks either "Like" or "Dislike".

        Returns:
            torch.Tensor: A tensor of feedback values with +1 for liked songs and -1 for disliked songs.
        """
        if self._playlist is None:
            raise ValueError("Playlist not set.")
        
        if self._playlist == self._prev_playlist:
            feedbacks = self._feedbacks
            return torch.tensor(feedbacks, dtype=float)

        # Import here so that if this method is not used, dependencies are not required.
        import tkinter as tk
        import pygame

        # Initialize pygame mixer for audio playback.
        pygame.mixer.init()

        feedbacks = []
        song_index = 0

        # Create the main window.
        root = tk.Tk()
        root.title("Music Feedback")
        root.geometry("500x200")

        # Label to display the current song.
        song_label = tk.Label(root, text="", font=("Arial", 14))
        song_label.pack(pady=20)

        # Function to load and play a song at a given index.
        def play_song(index: int):
            if index < len(self._playlist):
                song_path = self._playlist[index]
                song_label.config(text=f"Now playing:\n{song_path}")
                try:
                    pygame.mixer.music.load(song_path)
                    pygame.mixer.music.play()
                except Exception as e:
                    # If audio fails to play, show the error in the label.
                    song_label.config(text=f"Error playing {song_path}:\n{e}")
            else:
                song_label.config(text="No more songs.")

        # Callback when the user provides feedback.
        def on_feedback(is_like: bool):
            nonlocal song_index
            # Record the feedback: +1 for like, -1 for dislike.
            feedbacks.append(1 if is_like else -1)
            # Stop current song playback.
            pygame.mixer.music.stop()
            song_index += 1
            if song_index < len(self._playlist):
                play_song(song_index)
            else:
                # All songs processed: close the window.
                root.destroy()

        # Create the "Like" and "Dislike" buttons.
        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        like_button = tk.Button(
            button_frame,
            text="Like",
            command=lambda: on_feedback(True),
            width=10,
            height=2,
            bg="lightgreen"
        )
        like_button.pack(side="left", padx=20)

        dislike_button = tk.Button(
            button_frame,
            text="Dislike",
            command=lambda: on_feedback(False),
            width=10,
            height=2,
            bg="tomato"
        )
        dislike_button.pack(side="right", padx=20)
        
        # button to replay the song
        replay_button = tk.Button(
            button_frame,
            text="Replay",
            command=lambda: play_song(song_index),
            width=10,
            height=2,
            bg="lightblue"
        )
        replay_button.pack(side="right", padx=20)

        # Start by playing the first song.
        play_song(song_index)

        # Run the Tkinter event loop (this call is blocking until the window is closed).
        root.mainloop()

        # Clean up the mixer after finishing.
        pygame.mixer.quit()
        
        self._feedbacks = feedbacks
        self._prev_playlist = self._playlist

        # Convert feedback list to a PyTorch tensor.
        return torch.tensor(feedbacks, dtype=float)
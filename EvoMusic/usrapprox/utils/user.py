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
        Launch a two-pane GUI for evaluating songs in the user's playlist:
        • Left pane: a scrollable list of songs, clickable to play, arrow keys to move up/down.
        • Right pane: a logo, "Now playing" info, waveform display, playback controls,
        and a Finish button.

        Returns:
            torch.Tensor: a 1D tensor with +1/-1/0 for each song (like / dislike / unrated).
        """
        import tkinter as tk
        import pygame
        from PIL import Image, ImageTk
        import os
        import random
        import torch

        # If no playlist is set, raise an error.
        if self._playlist is None:
            raise ValueError("Playlist not set.")

        # If this playlist has been evaluated before, just return the stored feedbacks.
        if self._playlist == self._prev_playlist:
            return torch.tensor(self._feedbacks, dtype=torch.float)

        # Initialize the pygame mixer for audio playback.
        pygame.mixer.init()

        # Initialize feedback storage (0 = unrated, 1 = liked, -1 = disliked).
        feedbacks = [0] * len(self._playlist)

        # Which song index is currently playing/selected, or None if none.
        current_index = None

        # Keep references to the GUI elements for each song to update colors, etc.
        song_items = {}

        # -------- COLOR & FONT DEFINITIONS (Spotify-like) --------
        # Spotify-inspired color palette
        BG_COLOR = "#121212"       # main window background
        SIDEBAR_COLOR = "#121212"  # left pane background
        ITEM_COLOR = "#181818"     # unselected item background
        TEXT_COLOR = "#b3b3b3"     # standard text color
        SELECTED_COLOR = "#282828" # highlight for selected item (slightly lighter)
        ACCENT_COLOR = "#1DB954"   # primary accent (Spotify green)
        DISLIKE_COLOR = "#e91429"  # red for dislike

        TITLE_FONT = ("Helvetica", 16, "bold")
        LABEL_FONT = ("Helvetica", 12)
        NOW_PLAYING_FONT = ("Helvetica", 14)

        # Create the main window.
        root = tk.Tk()
        root.title("Music Feedback")
        root.geometry("1000x600")
        root.configure(bg=BG_COLOR)

        # --- LEFT PANE: Scrollable Playlist ---
        left_frame = tk.Frame(root, bg=SIDEBAR_COLOR, width=300)
        left_frame.pack(side="left", fill="y")

        playlist_header = tk.Label(
            left_frame,
            text="Playlist",
            bg=SIDEBAR_COLOR,
            fg=TEXT_COLOR,
            font=TITLE_FONT
        )
        playlist_header.pack(pady=10)

        playlist_canvas = tk.Canvas(left_frame, bg=SIDEBAR_COLOR, highlightthickness=0)
        playlist_canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=playlist_canvas.yview)
        scrollbar.pack(side="right", fill="y")
        playlist_canvas.configure(yscrollcommand=scrollbar.set)

        playlist_container = tk.Frame(playlist_canvas, bg=SIDEBAR_COLOR)
        playlist_canvas.create_window((0, 0), window=playlist_container, anchor="nw")

        # Ensure the canvas scroll region updates when items change size.
        def on_container_resize(event):
            playlist_canvas.configure(scrollregion=playlist_canvas.bbox("all"))
        playlist_container.bind("<Configure>", on_container_resize)

        # --- RIGHT PANE: Logo, "Now playing", waveform, and controls ---
        right_frame = tk.Frame(root, bg=BG_COLOR)
        right_frame.pack(side="right", fill="both", expand=True)

        # Load the JPEG logo using LANCZOS for resizing (Pillow 9+).
        try:
            with Image.open("img/logo_cropped.jpeg") as logo_image:
                logo_image = logo_image.resize((150, 150), Image.Resampling.LANCZOS)
                logo_photo = ImageTk.PhotoImage(logo_image)
            logo_label = tk.Label(right_frame, image=logo_photo, bg=BG_COLOR)
            logo_label.image = logo_photo  # store a reference so it doesn't get GC'd
            logo_label.pack(pady=10)
        except Exception as e:
            print(f"Error loading logo: {e}")

        current_song_label = tk.Label(
            right_frame,
            text="",
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            font=NOW_PLAYING_FONT
        )
        current_song_label.pack(pady=5)

        # A canvas for a dummy waveform display.
        waveform_canvas = tk.Canvas(right_frame, bg="#101010", height=200, highlightthickness=0)
        waveform_canvas.pack(pady=10, padx=10, fill="x")

        # A row of controls (Play/Pause/Replay) plus an evaluation label (heart or X).
        controls_frame = tk.Frame(right_frame, bg=BG_COLOR)
        controls_frame.pack(pady=10)

        play_button = tk.Button(
            controls_frame,
            text="Play",
            bg=ACCENT_COLOR,
            fg="#ffffff",
            relief="flat",
            width=10,
            command=lambda: play_current_song()
        )
        play_button.pack(side="left", padx=5)

        pause_button = tk.Button(
            controls_frame,
            text="Pause",
            bg=ACCENT_COLOR,
            fg="#ffffff",
            relief="flat",
            width=10,
            command=lambda: pause_song()
        )
        pause_button.pack(side="left", padx=5)

        replay_button = tk.Button(
            controls_frame,
            text="Replay",
            bg=ACCENT_COLOR,
            fg="#ffffff",
            relief="flat",
            width=10,
            command=lambda: replay_song()
        )
        replay_button.pack(side="left", padx=5)

        evaluation_label = tk.Label(
            controls_frame,
            text="",
            bg=BG_COLOR,
            fg=ACCENT_COLOR,
            font=("Helvetica", 16)
        )
        evaluation_label.pack(side="left", padx=10)

        finish_button = tk.Button(
            right_frame,
            text="Finish",
            bg=ACCENT_COLOR,
            fg="#ffffff",
            relief="flat",
            width=15,
            command=lambda: finish()
        )
        finish_button.pack(pady=10)

        # -------------- HELPER FUNCTIONS --------------

        def draw_waveform():
            """Draws a random 'waveform' on the waveform canvas for visual effect."""
            waveform_canvas.delete("all")
            width = waveform_canvas.winfo_width() or 400
            height = waveform_canvas.winfo_height() or 200
            num_points = 80
            step = width / max(1, (num_points - 1))
            points = [random.randint(0, height) for _ in range(num_points)]

            for i in range(num_points - 1):
                x1 = i * step
                y1 = height - points[i]
                x2 = (i + 1) * step
                y2 = height - points[i + 1]
                waveform_canvas.create_line(x1, y1, x2, y2, fill=ACCENT_COLOR, width=2)

        def play_song(index: int):
            """Play the song at the given index, update highlights and labels."""
            nonlocal current_index
            old_index = current_index
            current_index = index

            song_path = self._playlist[index]
            current_song_label.config(text=f"Now playing: {os.path.basename(song_path)}")

            # Un-highlight the old selection
            if old_index is not None and old_index != index:
                set_item_color(old_index, selected=False)

            # Highlight the new selection
            set_item_color(index, selected=True)

            # Load and play
            try:
                pygame.mixer.music.load(song_path)
                pygame.mixer.music.play()
            except Exception as e:
                current_song_label.config(text=f"Error playing {os.path.basename(song_path)}: {e}")

            draw_waveform()
            update_evaluation_label()

        def play_current_song():
            """Resume playback of the current track if paused."""
            if current_index is not None:
                pygame.mixer.music.unpause()

        def pause_song():
            """Pause the current track."""
            pygame.mixer.music.pause()

        def replay_song():
            """Rewind and replay the current track."""
            if current_index is not None:
                pygame.mixer.music.rewind()
                pygame.mixer.music.play()
                draw_waveform()
                update_evaluation_label()

        def finish():
            """Stop playback and close the app."""
            pygame.mixer.music.stop()
            root.destroy()

        def update_evaluation_label():
            """Refresh the heart or X beside the play/pause/replay buttons."""
            if current_index is None:
                evaluation_label.config(text="")
                return
            rating = feedbacks[current_index]
            if rating == 1:
                evaluation_label.config(text="❤", fg=ACCENT_COLOR)
            elif rating == -1:
                evaluation_label.config(text="✖", fg=DISLIKE_COLOR)
            else:
                evaluation_label.config(text="", fg="#ffffff")

        # --- Song Item Coloring & Rating ---

        def set_item_color(idx: int, selected: bool = False):
            """Set the color of the item in the playlist based on rating and selection."""
            rating = feedbacks[idx]
            if rating == 1:
                color = ACCENT_COLOR    # green for liked
            elif rating == -1:
                color = DISLIKE_COLOR   # red for disliked
            else:
                color = ITEM_COLOR      # neutral dark

            frame = song_items[idx]["frame"]
            label = song_items[idx]["label"]

            # If selected, highlight with a slightly lighter background
            if selected:
                frame.config(bg=SELECTED_COLOR)
                label.config(bg=SELECTED_COLOR)
            else:
                frame.config(bg=color)
                label.config(bg=color)

        def rate_song(idx: int, rating: int):
            """Record the rating and recolor the item. Update the evaluation label if it's the current track."""
            feedbacks[idx] = rating
            set_item_color(idx, selected=(idx == current_index))
            if idx == current_index:
                update_evaluation_label()

        # --- ARROW KEYS NAVIGATION ---

        def select_previous_song(event=None):
            """Move up in the playlist (arrow up)."""
            nonlocal current_index
            if not self._playlist:
                return
            if current_index is None:
                current_index = 0
            else:
                current_index = max(0, current_index - 1)
            play_song(current_index)
            scroll_to_item(current_index)

        def select_next_song(event=None):
            """Move down in the playlist (arrow down)."""
            nonlocal current_index
            if not self._playlist:
                return
            if current_index is None:
                current_index = 0
            else:
                current_index = min(len(self._playlist) - 1, current_index + 1)
            play_song(current_index)
            scroll_to_item(current_index)

        root.bind("<Up>", select_previous_song)
        root.bind("<Down>", select_next_song)

        def scroll_to_item(idx: int):
            """Scroll the left pane so item idx is in view."""
            if len(self._playlist) > 1:
                fraction = idx / float(len(self._playlist) - 1)
                playlist_canvas.yview_moveto(fraction)

        # --- BUILD THE PLAYLIST UI ---
        for idx, song_path in enumerate(self._playlist):
            item_frame = tk.Frame(playlist_container, bg=ITEM_COLOR, padx=5, pady=5)
            item_frame.pack(fill="x", pady=2, padx=5)

            song_name = os.path.basename(song_path)
            label = tk.Label(
                item_frame,
                text=song_name,
                bg=ITEM_COLOR,
                fg=TEXT_COLOR,
                font=LABEL_FONT,
                anchor="w"
            )
            label.pack(side="left", fill="x", expand=True)

            # Clicking the song name plays it immediately.
            label.bind("<Button-1>", lambda e, i=idx: play_song(i))

            # Dislike (✖) button
            dislike_button = tk.Button(
                item_frame,
                text="✖",
                bg=ITEM_COLOR,
                fg=TEXT_COLOR,
                relief="flat",
                command=lambda i=idx: rate_song(i, -1)
            )
            dislike_button.pack(side="right", padx=2)

            # Like (❤) button
            like_button = tk.Button(
                item_frame,
                text="❤",
                bg=ITEM_COLOR,
                fg=TEXT_COLOR,
                relief="flat",
                command=lambda i=idx: rate_song(i, 1)
            )
            like_button.pack(side="right", padx=2)

            song_items[idx] = {"frame": item_frame, "label": label}

        # If there are songs, start playing the first one.
        if self._playlist:
            play_song(0)

        # Start the Tkinter event loop.
        root.mainloop()

        # Cleanup the mixer.
        pygame.mixer.quit()

        # Save and store feedback in the user object.
        self._feedbacks = feedbacks
        self._prev_playlist = self._playlist

        # Return the final feedback as a tensor.
        return torch.tensor(feedbacks, dtype=torch.float)

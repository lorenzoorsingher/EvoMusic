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

        self._train_dataloader = None
        self._test_dataloader = None

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

    def add_to_memory(self, batch: torch.Tensor):
        self._memory.add_to_memory(batch)

    def empty_memory(self):
        self._memory.empty_memory()

    def set_dataloaders(
        self, train_dataloader: DataLoader, test_dataloader: DataLoader
    ):
        """
        set the dataloaders.
        """
        if self._train_dataloader is not None or self._test_dataloader is not None:
            raise ValueError("Dataloaders already set.")

        self._train_dataloader = train_dataloader
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
        return self._train_dataloader

    @property
    def test_dataloader(self) -> DataLoader:
        """
        get the test dataloader.
        """
        return self._test_dataloader


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

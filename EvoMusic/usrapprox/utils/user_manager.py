import json
import torch
from torch.utils.data import DataLoader

from EvoMusic.usrapprox.models.usr_emb import UsrEmb
from EvoMusic.configuration import AlignerV2Config, TrainConfig, UserConfig

from EvoMusic.usrapprox.utils.dataset import (
    ContrDatasetWrapper,
    UserDefinedContrastiveDataset,
)
from EvoMusic.usrapprox.utils.user import RealUser, SynthUser, User


class UserManager:
    def __init__(
        self,
        users: list[RealUser, SynthUser],
        users_config: UserConfig,
        user_delta: int,
        aligner_config: AlignerV2Config = AlignerV2Config(),
        device: str = "cuda",
    ):

        dict_users = {}

        for i, user in enumerate(users):
            if isinstance(user, RealUser):
                user.set_memory_size(users_config.memory_length)
                user.set_minibatch(users_config.minibatch)
                user.set_user_id(user_delta + i + 1)
                dict_users[user.uuid] = user

            elif isinstance(user, SynthUser):
                user.set_memory_size(users_config.memory_length)
                user.set_minibatch(users_config.minibatch)
                user.set_user_id(user_delta + i + 1)
                user.set_user_reference(user.uuid)
                dict_users[user.uuid] = user

            else:
                raise ValueError("User type not recognized.")

        self._users = dict_users

        self.device = device
        self.usr_emb = UsrEmb(
            users_config=users_config, aligner_config=aligner_config, device=device
        )

        self.usr_emb.eval()
        self.usr_emb.to(self.device)

    def __getitem__(self, uuid: int) -> User:
        if uuid not in self._users:
            print(self._users.keys())
            raise ValueError(f"User {uuid} not found.")
        return self._users[uuid]

    def update_memory(self, user: User, batch: torch.Tensor, feedback: torch.Tensor):
        user.add_to_memory(batch, feedback)

    def get_memory(self, user: User):
        return user.memory

    def clear_memory(self, user: User):
        user.empty_memory()

    def set_memory_device(self, user: User, device: torch.device):
        user.set_memory_device(device)

    def user_step(self, user: User, batch: torch.Tensor):
        if isinstance(user, RealUser) or isinstance(user, SynthUser):
            user_embedding, embeddings, temperature, music_score = self.usr_emb(
                user.user_id, batch
            )

        else:
            raise ValueError("User type not recognized.")

        return user_embedding, embeddings, temperature, music_score

    def feedback_step(self, user: User, batch: torch.Tensor):
        """
        This works with torch.no_grad().
        """
        if isinstance(user, RealUser):
            raise NotImplementedError()
        elif isinstance(user, SynthUser):
            with torch.no_grad():
                # usr_embedding_id = user._user_id

                user_embedding, embeddings, temperature, music_score = self.usr_emb(
                    user.model_reference_id, batch
                )
        else:
            raise ValueError("User type not recognized.")

        return user_embedding, embeddings, temperature, music_score

    def load_dataset(
        self, user: User, train_config: TrainConfig, split: str
    ) -> DataLoader:
        """
        Load the training and testing datasets for a user.

        Args:
            user (User): The user to load the datasets for.
            train_config (TrainConfig): The training configuration.
            split (str): The split to load the dataset for, either "train" or "test".
        """
        assert split in ["train", "test"]

        if train_config.type == "ContrDatasetMERT":
            print("Using ContrDatasetMERT")
            with open(train_config.splits_path, "r") as f:
                splits = json.load(f)

            dataset = ContrDatasetWrapper(
                train_config.embs_path,
                train_config.stats_path,
                split=splits[split],
                usrs=user.user_id,
                nneg=train_config.nneg,
                multiplier=train_config.multiplier,
            )

        else:
            print("Using UserDefinedContrastiveDataset")
            dataset = UserDefinedContrastiveDataset(
                alignerV2=self.usr_emb,
                splits_path=train_config.splits_path,
                embs_path=train_config.embs_path,
                user_id=user.user_id,
                npos=train_config.npos,
                nneg=train_config.nneg,
                batch_size=train_config.batch_size,
                num_workers=train_config.num_workers,
                partition="train",
                random_pool=train_config.random_pool,
            )

        shuffle = False
        if split == "train":
            shuffle = True

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=shuffle,
            num_workers=train_config.num_workers,
        )

        if split == "train":
            user.set_train_dataloader(dataloader)
        else:
            user.set_test_dataloader(dataloader)

        return dataloader

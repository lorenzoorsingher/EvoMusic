"""
Class and methods for user management and training.
"""

import json
from torch.utils.tensorboard import SummaryWriter


import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from usrapprox.usrapprox.models.aligner_v2 import AlignerV2Wrapper

# from usrapprox.usrapprox.models.probabilistic import probabilistic_model_torch
from usrapprox.usrapprox.models.usr_emb import UsrEmb
from usrapprox.usrapprox.utils.config import AlignerV2Config, UserConfig, TrainConfig
from usrapprox.usrapprox.utils.dataset import (
    UserDefinedContrastiveDataset,
    ContrDatasetWrapper,
)
from usrapprox.usrapprox.utils.user import RealUser, SynthUser, User
from usrapprox.usrapprox.utils.utils import ScoreToFeedback


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
                user.set_user_id(user_delta + i + 1)
                dict_users[user.uuid] = user

            elif isinstance(user, SynthUser):
                user.set_memory_size(users_config.memory_length)
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

    def load_datasets(
        self, user: User, train_config: TrainConfig
    ) -> tuple[DataLoader, DataLoader]:
        """
        This method is used to load the datasets.
        """
        if user.train_dataloader is None and user.test_dataloader is None:
            if train_config.type == "ContrDatasetMERT":
                print("Using ContrDatasetMERT")
                with open(train_config.splits_path, "r") as f:
                    splits = json.load(f)

                train_dataset = ContrDatasetWrapper(
                    train_config.embs_path,
                    train_config.stats_path,
                    split=splits["train"],
                    usrs=user.user_id,
                    nneg=train_config.nneg,
                    multiplier=train_config.multiplier,
                )

                test_dataset = ContrDatasetWrapper(
                    train_config.embs_path,
                    train_config.stats_path,
                    split=splits["test"],
                    usrs=user.user_id,
                    nneg=train_config.nneg,
                    multiplier=train_config.multiplier,
                )
            else:
                print("Using UserDefinedContrastiveDataset")
                train_dataset = UserDefinedContrastiveDataset(
                    alignerV2=self.usr_emb,
                    splits_path=train_config.splits_path,
                    embs_path=train_config.embs_path,
                    user_id=user.user_id,
                    npos=train_config.npos,
                    nneg=train_config.nneg,
                    batch_size=train_config.batch_size,
                    num_workers=train_config.num_workers,
                    partition="train",
                )

                test_dataset = UserDefinedContrastiveDataset(
                    alignerV2=self.usr_emb,
                    splits_path=train_config.splits_path,
                    embs_path=train_config.embs_path,
                    user_id=user.user_id,
                    npos=train_config.npos,
                    nneg=train_config.nneg,
                    batch_size=train_config.batch_size,
                    num_workers=train_config.num_workers,
                    partition="test",
                )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=train_config.batch_size,
                shuffle=True,
                num_workers=train_config.num_workers,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=train_config.batch_size,
                shuffle=True,
                num_workers=train_config.num_workers,
            )

            user.set_dataloaders(train_dataloader, test_dataloader)

        return user.train_dataloader, user.test_dataloader

    def update_memory(self, user: User, batch: torch.Tensor):
        user.add_to_memory(batch)

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


class UsersTrainManager:
    def __init__(
        self,
        users_config: UserConfig,
        writer: SummaryWriter,
        aligner_config: AlignerV2Config = AlignerV2Config(),
        device: str = "cuda",
    ):
        self.writer = writer

        self.aligner_config = aligner_config
        self.device = device

        # Create the base UsrEmb model
        self.usr_emb = UsrEmb(users_config, aligner_config, device)

        self._alignerv2 = None

        # Create the users
        self._users = {
            user_id: User(
                user_id,
                self.usr_emb.get_user_embedding_weights,
                users_config.memory_length,
            )
            for user_id in users_config.user_ids
        }

        self._last_used_user = None
        self._score_to_feedback = ScoreToFeedback(self.device)


    # infonce - nxtent

    def __feedback_loss(self, music_scores, target_feedback, temperature):
        """
        Computes the InfoNCE loss.

        Args:
            music_scores (Tensor): Shape (batch_size, num_songs), raw scores for each song.
            target_feedback (Tensor): Shape (batch_size, num_songs), values in {-1, 1}.
            temperature (float): Temperature scaling parameter.

        Returns:
            Tensor: Scalar loss value.
        """
        # Mask positive and negative feedback
        positive_mask = (target_feedback == 1).float()  # Shape (batch_size, num_songs)
        # negative_mask = (target_feedback == -1).float()  # Shape (batch_size, num_songs)

        # Scale scores with temperature
        scaled_scores = music_scores * torch.exp(temperature)

        # Compute numerator: sum of exponentials of positive scores
        # positive_scores = scaled_scores
        positive_exp = torch.exp(scaled_scores) * positive_mask
        positive_sum = positive_exp.sum(dim=1)  # Shape (batch_size,)

        # Compute denominator: sum of exponentials of all scores (mask ensures only valid feedback)
        all_mask = (target_feedback != 0).float()  # Mask valid scores
        all_exp = torch.exp(scaled_scores) * all_mask
        all_sum = all_exp.sum(dim=1)  # Shape (batch_size,)

        # Avoid division by zero with a small epsilon
        epsilon = 1e-6

        # Compute InfoNCE loss
        info_nce_loss = -torch.log(
            (positive_sum + epsilon) / (all_sum + epsilon)
        ).mean()

        return info_nce_loss

    def __load_datasets(self, user: User) -> tuple[DataLoader, DataLoader]:
        """
        This method is used to load the datasets.
        """
        if user.train_dataloader is None and user.test_dataloader is None:
            if user.train_config.type == "ContrDatasetMERT":
                print("Using ContrDatasetMERT")
                with open(user.train_config.splits_path, "r") as f:
                    splits = json.load(f)

                train_dataset = ContrDatasetWrapper(
                    user.train_config.embs_path,
                    user.train_config.stats_path,
                    split=splits["train"],
                    usrs=user.user_id,
                    nneg=user.train_config.nneg,
                    multiplier=user.train_config.multiplier,
                )

                test_dataset = ContrDatasetWrapper(
                    user.train_config.embs_path,
                    user.train_config.stats_path,
                    split=splits["test"],
                    usrs=user.user_id,
                    nneg=user.train_config.nneg,
                    multiplier=user.train_config.multiplier,
                )
            else:
                print("Using UserDefinedContrastiveDataset")
                train_dataset = UserDefinedContrastiveDataset(
                    alignerV2=self._alignerv2,
                    splits_path=user.train_config.splits_path,
                    embs_path=user.train_config.embs_path,
                    user_id=user.user_id,
                    npos=user.train_config.npos,
                    nneg=user.train_config.nneg,
                    batch_size=user.train_config.batch_size,
                    num_workers=user.train_config.num_workers,
                    partition="train",
                )

                test_dataset = UserDefinedContrastiveDataset(
                    alignerV2=self._alignerv2,
                    splits_path=user.train_config.splits_path,
                    embs_path=user.train_config.embs_path,
                    user_id=user.user_id,
                    npos=user.train_config.npos,
                    nneg=user.train_config.nneg,
                    batch_size=user.train_config.batch_size,
                    num_workers=user.train_config.num_workers,
                    partition="test",
                )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=user.train_config.batch_size,
                shuffle=True,
                num_workers=user.train_config.num_workers,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=user.train_config.batch_size,
                shuffle=True,
                num_workers=user.train_config.num_workers,
            )

            user.set_dataloaders(train_dataloader, test_dataloader)

        return user.train_dataloader, user.test_dataloader

    def __check_user(self, user_id: int) -> None:
        """
        This method is used to check if the user is in the users.
        """
        if user_id not in self._users:
            raise ValueError(f"User {user_id} not found.")

    def __get_user(self, user_id: int) -> User:
        """
        This method is used to get the user.
        """
        self.__check_user(user_id)
        return self._users[user_id]

    def set_train_config(
        self, user_id: int, train_config: TrainConfig = TrainConfig()
    ) -> None:
        """
        This method is used to set the training configuration for a user.
        """
        user = self.__get_user(user_id)
        user.set_train_config(train_config)

    def finetune(self):
        # TODO: add here the memory part

        raise NotImplementedError()

    def __train(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        user: User,
        epoch: int,
    ):
        self.usr_emb.eval()
        # self.usr_emb.users.train()

        losses = []
        user1 = self.__get_user(1)

        for i, (tracks) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            # tracks = torch.cat((posemb, negemb), dim=1)
            # self.__set_model_embedding(user1)
            # self.__set_model_embedding(user)

            tracks = tracks.to(self.device)
            _, _, temperature, music_score = self.usr_emb(tracks)

            ids_aligner = torch.LongTensor([user.user_id] * tracks.shape[0]).to(
                self.device
            )
            _, _, _, target_score = self._alignerv2(ids_aligner, tracks)

            target_feedback = self._score_to_feedback.get_feedback(target_score).to(
                self.device
            )

            loss = self.__feedback_loss(music_score, target_feedback, temperature)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.usr_emb.parameters(), 5)
            optimizer.step()

        self.writer.add_scalar(
            "Loss/Training", torch.tensor(losses).mean().item(), epoch
        )

    def __eval(self, val_loader: DataLoader, user: User, epoch: int):
        self.usr_emb.eval()
        losses = []

        # Define empty torch tensors
        music_scores = torch.empty(0).to(self.device)
        target_scores = torch.empty(0).to(self.device)

        with torch.no_grad():
            for i, (tracks) in enumerate(
                tqdm(val_loader, desc="Evaluating", leave=False)
            ):
                # tracks = torch.cat((posemb, negemb), dim=1)
                tracks = tracks.to(self.device)

                # Get "actions" from usr_emb and aligner
                _, _, temperature, music_score = self.usr_emb(tracks)

                ids_aligner = torch.LongTensor([user.user_id] * tracks.shape[0]).to(
                    self.device
                )
                _, _, _, target_score = self._alignerv2(ids_aligner, tracks)

                # Calculate loss
                target_feedback = self._score_to_feedback.get_feedback(target_score)
                loss = self.__feedback_loss(music_score, target_feedback, temperature)
                losses.append(loss.item())

                music_scores = torch.cat((music_scores, music_score))
                target_scores = torch.cat((target_scores, target_score))

        # Access the weights of the user embeddings
        usr_emb_weight = self.usr_emb.users.weight  # Shape: [1, EMB_SIZE]
        aligner_weight = self._alignerv2.users.weight[user.user_id]  # Shape: [EMB_SIZE]

        # Expand aligner_weight to match the dimensions of usr_emb_weight
        aligner_weight = aligner_weight.unsqueeze(0)  # Shape: [1, EMB_SIZE]

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            usr_emb_weight, aligner_weight, dim=-1
        )

        average_cosine_similarity_on_model = cosine_sim.mean().item()

        losses = torch.tensor(losses).mean().item()

        mse = torch.nn.functional.cosine_similarity(
            torch.Tensor(music_scores), torch.Tensor(target_scores)
        ).mean()

        self.writer.add_scalar("Loss/Validation", losses, epoch)
        self.writer.add_scalar(
            "Validation/Cosine Model", average_cosine_similarity_on_model, epoch
        )
        self.writer.add_scalar("Validation/Cosine Scores", mse, epoch)

    def test_training(self, user_id: int):
        ### load alignerV2Wrapper
        # Set the alignerv2 if it is not set
        self.__set_alignerv2()

        ### Get the user
        user = self.__get_user(user_id)

        # load dataset
        train_dataloader, test_dataloader = self.__load_datasets(user)

        # load the most recent user embedding for the current user
        self.__set_model_embedding(user)

        # optimizer
        optimizer = torch.optim.AdamW(
            self.usr_emb.users.parameters(), lr=user.train_config.lr
        )

        # at the end of each epoch update the user embedding
        for epoch in tqdm(
            range(user.train_config.epochs),
            desc=f"Training user {user_id}",
        ):
            self.__train(train_dataloader, optimizer, user, epoch)

            self.__eval(test_dataloader, user, epoch)

        # save the user embedding
        self.__update_user_embedding(user)

    # def get_user_score(self, user_id: int, batch: torch.Tensor):
    #     """
    #     This method is used to get the score of a user given a batch.
    #     TODO: aggiungere per farlo andare col modello di Lollo
    #     """
    #     self.__check_user(user_id)
    #     self.__set_model_embedding(user_id)

    #     _, _, score = self.usr_emb(batch)
    #     normalized_similarities = (score + 1) / 2

    #     feedback_wrt_song = probabilistic_model_torch(normalized_similarities)

    #     return feedback_wrt_song

    def save_weights(self):
        raise NotImplementedError()

    def load_weights(self):
        raise NotImplementedError()

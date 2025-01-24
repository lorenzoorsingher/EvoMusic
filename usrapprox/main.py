import json
from tqdm import tqdm
from EvoMusic.usrapprox.utils.config import TrainConfig, UserConfig
from EvoMusic.usrapprox.utils.user import RealUser, SynthUser, User
from EvoMusic.usrapprox.utils.user_manager import UserManager
from EvoMusic.usrapprox.utils.user_train_manager import UsersTrainManager

import torch
from torch.utils.data import DataLoader

# seed and deterministic
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# set tensorboard
from torch.utils.tensorboard import SummaryWriter

def load_datasets(
        user_manager: UserManager, user: User, train_config: TrainConfig
    ) -> tuple[DataLoader, DataLoader]:
        """
        This method is used to load the datasets.
        """
        from dataset import (
            UserDefinedContrastiveDataset,
            ContrDatasetWrapper,
        )

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
                    alignerV2=user_manager.usr_emb,
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
                    alignerV2=user_manager.usr_emb,
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

def train(
        user_train_manager: UsersTrainManager,
        train_loader: torch.utils.data.DataLoader,
        user: User,
        epoch: int,
    ):
        losses = []

        user_train_manager._user_manager.clear_memory(user)

        user_train_manager._user_manager.usr_emb.eval()
        for tracks in tqdm(train_loader, desc="Training", leave=False):
            user_train_manager._user_manager.update_memory(user, tracks)
            tracks = user_train_manager._user_manager.get_memory(user)
            tracks = tracks.to(user_train_manager.device)

            loss = user_train_manager.train_one_step(tracks, user)
            losses.append(loss)

        user_train_manager.writer.add_scalar(
            "Loss/Training", torch.tensor(losses).mean().item(), epoch
        )

def test_train(
        user_train_manager: UsersTrainManager,
        user: User,
    ):
        train_dataloader, test_dataloader = load_datasets(
            user_train_manager._user_manager, user, user_train_manager._train_config
        )

        user_train_manager.set_optimizer()

        for epoch in tqdm(
            range(user_train_manager._train_config.epochs),
            desc=f"Training user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}",
        ):
            train(user_train_manager, train_dataloader, user, epoch)

            user_train_manager.eval(test_dataloader, user, epoch)

if __name__ == "__main__":
    writer = SummaryWriter()

    user = 0

    # Give me a list of users (initialize only IDs)
    # users = [RealUser(0), SynthUser(1)]
    users = [
        SynthUser(user),
        SynthUser(user + 1),
        SynthUser(user + 5),
    ]  # , SynthUser(1)]

    user_config = UserConfig(memory_length=1, amount=len(users))

    user_train_config = TrainConfig(
        batch_size=50, npos=15, nneg=15, epochs=20, num_workers=6, lr=0.001
    )

    # torch get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manager = UsersTrainManager(
        users=users,
        users_config=user_config,
        train_config=user_train_config,
        writer=writer,
        device=device,
    )

    user1 = manager.get_user(user)

    test_train(manager, user1)
    

    writer.close()

from tqdm import tqdm
from EvoMusic.configuration import TrainConfig, UserConfig
from EvoMusic.usrapprox.utils.user import SynthUser, User
from EvoMusic.usrapprox.utils.user_train_manager import UsersTrainManager

import torch

import wandb

# seed and deterministic
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

def train(
        user_train_manager: UsersTrainManager,
        train_loader: torch.utils.data.DataLoader,
        user: User,
        epoch: int,
    ):
        losses = []

        # user_train_manager._user_manager.clear_memory(user)

        user_train_manager._user_manager.usr_emb.eval()
        for tracks in tqdm(train_loader, desc="Training", leave=False):
            tracks = tracks.to(user_train_manager.device)
            user_train_manager.update_memory(user, tracks)
            user_train_manager.set_memory_device(user, user_train_manager.device)
            tracks, feedbacks = user_train_manager.get_memory(user)
            # tracks = tracks.to(user_train_manager.device)
            # feedbacks = feedbacks.to(user_train_manager.device)
            # user_train_manager._user_manager.usr_emb.to(user_train_manager.device)

            loss = user_train_manager.train_one_step(tracks, feedbacks, user)
            
            losses.append(loss)

        wandb.log(
            {"Loss/Training": torch.tensor(losses).mean().item()}, step=epoch
        )

def test_train(
        user_train_manager: UsersTrainManager,
        user: User,
    ):
        train_dataloader = user_train_manager._user_manager.load_dataset(user, user_train_manager._train_config, "train")
        test_dataloader = user_train_manager._user_manager.load_dataset(user, user_train_manager._train_config, "test")
        
        user_train_manager.set_optimizer()

        for epoch in tqdm(
            range(user_train_manager._train_config.epochs),
            desc=f"Training user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}",
        ):
            train(user_train_manager, train_dataloader, user, epoch)

            user_train_manager.eval(test_dataloader, user, epoch)

if __name__ == "__main__":
    wandb.init(
                project="test", 
                name="ea main test", 
    )

    user = 0

    # Give me a list of users (initialize only IDs)
    # users = [RealUser(0), SynthUser(1)]
    users = [
        SynthUser(user),
        SynthUser(user + 1),
        SynthUser(user + 5),
    ]  # , SynthUser(1)]

    user_config = UserConfig(memory_length=50, amount=len(users))

    user_train_config = TrainConfig(
        batch_size=1, npos=15, nneg=15, epochs=10, 
        # num_workers=6, 
        lr=0.001, #random_pool=20
    )

    # torch get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manager = UsersTrainManager(
        users=users,
        users_config=user_config,
        train_config=user_train_config,
        device=device,
    )

    user1 = manager.get_user(user)

    test_train(manager, user1)
from usrapprox.usrapprox.utils.config import TrainConfig, UserConfig
from usrapprox.usrapprox.utils.user import RealUser, SynthUser
from usrapprox.usrapprox.utils.user_train_manager import UsersTrainManager

import torch

# seed and deterministic
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# set tensorboard
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()

    user = 0

    # Give me a list of users (initialize only IDs)
    # users = [RealUser(0), SynthUser(1)]
    users = [SynthUser(user), SynthUser(user+1), SynthUser(user+5)]#, SynthUser(1)]


    user_config = UserConfig(memory_length=5, amount=len(users))

    user_train_config = TrainConfig(
        batch_size=50,
        npos=15,
        nneg=15,
        epochs=20,
        num_workers=6,
        lr=0.001
    )

    # torch get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manager = UsersTrainManager(
        users=users,
        users_config=user_config,
        writer=writer,
        device=device,
    )

    user1 = manager.get_user(user)

    manager.test_train(user1, user_train_config)


    writer.close()

from usrapprox.usrapprox.utils.config import TrainConfig, UserConfig
from usrapprox.usrapprox.utils.user_manager import UsersTrainManager

import torch

# seed and deterministic
torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True

# set tensorboard
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    writer = SummaryWriter()

    user = 0

    users = UserConfig(user_ids=[user], memory_length=5)
    user_train_config = TrainConfig(
        batch_size=30,
        npos=15,
        nneg=15,
        epochs=10,
        num_workers=6,
        type="ContrDatasetMERT",
    )

    manager = UsersTrainManager(users, writer=writer)

    manager.set_train_config(user, user_train_config)

    manager.test_training(user)

    writer.close()

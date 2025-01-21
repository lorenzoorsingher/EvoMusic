

from usrapprox.usrapprox.utils.users import UserConfig, UsersManager


if __name__ == "__main__":
    users = UserConfig(user_ids=[1], memory_length=5)

    manager = UsersManager(users)
    manager.set_train_config(1)

    manager.test_training(1)
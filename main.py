from usrapprox.usrapprox.utils.users import TrainConfig, UserConfig, UsersManager


if __name__ == "__main__":
    
    
    users = UserConfig(user_ids=[1], memory_length=5)
    train_config = {
        1: TrainConfig(
            batch_size=128,
            npos=5,
            nneg=5,
            epochs=50
        )
    }
    manager = UsersManager(users)
    
    manager.set_train_config(1)

    manager.test_training(1)
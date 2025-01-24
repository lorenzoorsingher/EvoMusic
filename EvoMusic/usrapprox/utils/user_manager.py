import torch

from EvoMusic.usrapprox.models.usr_emb import UsrEmb
from EvoMusic.configuration import AlignerV2Config, UserConfig

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

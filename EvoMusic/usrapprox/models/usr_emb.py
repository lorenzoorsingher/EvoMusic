import torch
import torch.nn as nn

from EvoMusic.configuration import AlignerV2Config, UserConfig
from EvoMusic.user_embs.model import AlignerV2


class UsrEmb(AlignerV2):
    """ """

    def __init__(
        self,
        users_config: UserConfig,
        aligner_config: AlignerV2Config = AlignerV2Config,
        device: str = "cuda:0",
    ):
        model_state, setup, _ = AlignerV2.load_model(aligner_config.abs_file_path)

        aligner_config = {**aligner_config.__dict__, **setup}

        super().__init__(
            n_users=aligner_config["nusers"] + users_config.amount + 1,
            emb_size=aligner_config["emb_size"],
            prj_size=aligner_config["prj_size"],
            prj_type=aligner_config["prj"],
            aggragation=aligner_config["aggr"],
            lt=aligner_config["learnable_temp"],
            temp=aligner_config["temp"],
            drop=aligner_config["drop"],
        )
        self._device = device

        users_embedding = model_state["users.weight"]

        # remove the user embedding from the model state
        del model_state["users.weight"]

        self.load_state_dict(model_state, strict=False)

        self.__set_default_embedding(
            users_embedding, users_config.init, users_config.rmean
        )  # which is an avg of the weights of the pre-trained model embeddings

        self.to(device)

    def forward(self, idx, batch):

        index = torch.LongTensor([idx for _ in range(batch.shape[0])])

        index = index.to(self._device)
        # super().to(self._device)

        user_embedding, embeddings, temperature = super().forward(index, batch)

        music_score = self.calculate_score(user_embedding, embeddings)

        return user_embedding, embeddings, temperature, music_score

    def __set_default_embedding(self, users_embedding: dict, init: str, rmean: int):
        """
        Set the user embedding of the model. It has three options:
        - random: the user embedding is set randomly
        - mean: the user embedding is set as the mean of the user embeddings
        - rmean: the user embedding is set as the mean of the user embeddings plus a random noise

        Args:
            users_embedding (dict): The user embedding to set.
            init (str): The initialization method.
            rmean (int): The random noise amount, only used for `rmean` initialization.

        """
        # users_embedding : [967, 256]
        if init == "random":
            user_embedding = torch.randn(users_embedding.shape[1]).to(users_embedding.device)
        elif init == "mean":
            user_embedding = users_embedding.mean(dim=0)  # [256]
            # user_embedding = users_embedding[0]
        elif init == "rmean":
            # is mean + random noise
            user_embedding = (
                users_embedding.mean(dim=0)
                + torch.randn(users_embedding.shape[1]).to(users_embedding.device) * rmean
            )

        # attach on dimension 0 `n` times average_suer_embedding to the user embedding
        # `n` is the number of extra users, given by users_embedding.shape[0] and
        # `self.users.weight.shape[0]` is the number of users in the model

        total_embedding = torch.cat(
            (
                users_embedding,
                user_embedding.repeat(
                    self.users.weight.shape[0] - users_embedding.shape[0], 1
                ),
            )
        )

        self.users.weight = nn.Parameter(total_embedding)

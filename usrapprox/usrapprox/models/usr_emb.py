import torch
import torch.nn as nn
import torch.nn.functional as F


from usrapprox.usrapprox.utils.config import AlignerV2Config
from usrembeds.models.model import AlignerV2


class UsrEmb(nn.Module):
    """
    This model is designed to behave as similar as possible to the original AlignerV2 model,
    but as we need a model that doesn't have a set of users embedding, we take the weights of
    the pre-trained AlignerV2 model and use them. The **main difference** is that the weights
    of the users embedding are averaged and in the forward methods no user id is passed as input,
    but directly the user embedding. This makes possible to train the model with a set of users
    embedding and then use the same model and weights to make predictions with a different set.
    """

    def __init__(
        self,
        config: AlignerV2Config = AlignerV2Config,
        device: str = "cuda:0",
        original_shape: bool = False,
    ):
        super().__init__()
        self.original_shape = original_shape

        # Get the embedding weights from AlignerV2, set the user embedding to the average of the weights
        self.user = nn.Embedding(1, config.emb_size)
        self.__set_default_embedding(config)
        self.default_embedding = self.user.weight.clone()
        self.last_embedding = self.default_embedding.clone()

        self.prj_size = config.prj_size + 1  # +1 for the feedback
        # self.aggragation = config.aggr
        self.noise_level = config.noise_level

        # Aggragation: gating-tanh
        self.gate = nn.Linear(self.prj_size * 13, self.prj_size * 13)
        self.gate_transform = nn.Linear(self.prj_size, self.prj_size)
        self.gate_norm = nn.LayerNorm(self.prj_size)
        self.gate_out_norm = nn.LayerNorm(self.prj_size)

        # global
        self.linear1 = nn.Linear(config.emb_size, config.hidden_size)
        self.fc1 = nn.Linear(config.emb_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, self.prj_size)
        self.fc2 = nn.Linear(config.hidden_size, self.prj_size)
        self.ln_usr = nn.LayerNorm(self.prj_size)
        self.linear3 = nn.Linear(config.emb_size, self.prj_size)

        self.drop = nn.Dropout(config.drop)

        # Prj-type: shared
        self.f5 = nn.Linear(self.prj_size, config.hidden_size)
        self.f6 = nn.Linear(config.hidden_size, self.prj_size)
        if self.original_shape:
            self.f7 = nn.Linear(self.prj_size, self.prj_size - 1)
            self.ln2 = nn.LayerNorm(self.prj_size - 1)
        else:
            self.ln2 = nn.LayerNorm(self.prj_size)

        self.to(device)

    def forward(
        self, music_combined: torch.Tensor, user_embedding: torch.Tensor | None = None
    ):

        if user_embedding is not None:
            self.set_user_embedding(user_embedding)
        else:
            self.set_user_embedding(self.last_embedding)

        usr_x = F.gelu(self.fc1(self.user.weight)) + self.linear1(self.user.weight)
        usr_x = (
            F.gelu(self.fc2(usr_x))
            + self.linear2(usr_x)
            + self.linear3(self.user.weight)
        )
        usr_x = self.ln_usr(usr_x)

        batch_size = music_combined.size(0)
        n_emb = music_combined.size(1)

        if self.training and self.noise_level > 0:
            music_combined = (
                music_combined + torch.randn_like(music_combined) * self.noise_level
            )

        # Aggragation: gating-tanh
        # [batch, NPOS+NNEG, 13, emb_size] -> [batch, NPOS+NNEG, 13*emb_size]
        music_embs_reshaped = music_combined.view(batch_size, n_emb, self.prj_size * 13)
        # compute weights for each dimension of the music embeddings to be normalized
        weights = self.gate(music_embs_reshaped).view(
            batch_size, n_emb, 13, self.prj_size
        )
        # normalize over the 13 dimensions
        weights = F.tanh(weights)
        # apply weighted sum to the music embeddings
        music_combined = self.gate_norm(
            music_combined + F.gelu(self.gate_transform(music_combined))
        )

        music_x = (weights * music_combined).sum(dim=2)
        music_x = self.gate_out_norm(music_x)

        # prj-type: shared
        music_x = self.f6(self.drop(F.gelu(self.f5(music_x)))) + music_x
        if self.original_shape:
            music_x = self.f7(music_x)
        music_x = self.ln2(music_x)

        usr_x = self.f6(self.drop(F.gelu(self.f5(usr_x)))) + usr_x
        if self.original_shape:
            usr_x = self.f7(usr_x)
        usr_x = self.ln2(usr_x)

        if self.training:
            if self.original_shape:
                drop_mask = (torch.rand(self.prj_size - 1) < self.drop.p).to(
                    music_x.device
                )
            else:
                drop_mask = (torch.rand(self.prj_size) < self.drop.p).to(music_x.device)

            music_x = music_x * drop_mask
            usr_x = usr_x * drop_mask

        # self.last_embedding = usr_x.clone()

        return usr_x, music_x

    def __set_default_embedding(self, config: AlignerV2Config) -> None:
        """
        This method is used to set the default user embedding.

        The default user embedding is set to the average of the weights of the pre-trained model.

        Args:
            config (AlignerV2Config): The configuration of the model.
        """
        starting_user_embedding = self.__get_user_embedding(config)
        self.set_user_embedding(starting_user_embedding.unsqueeze(0))

    def __get_user_embedding(self, config: AlignerV2Config) -> torch.Tensor:
        """
        This method is used to get the user embedding from the pre-trained model.

        The user embedding is averaged and returned.

        Args:
            config (AlignerV2Config): The configuration of the model.

        Returns:
            torch.Tensor: The averaged user embedding.
        """
        # Load the model
        model_state, _, _ = AlignerV2.load_model(config.abs_file_path)

        embedding_users_weight = model_state["users.weight"]
        average_user_embedding = embedding_users_weight.mean(dim=0)

        return average_user_embedding

    def set_user_embedding(self, user_embedding: torch.Tensor) -> None:
        """
        This method is used to set the user embedding of the model.

        Args:
            user_embedding (torch.Tensor): The user embedding to set.
        """
        self.user.weight = nn.Parameter(user_embedding)

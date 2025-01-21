import torch
import torch.nn as nn
import torch.nn.functional as F


from usrapprox.usrapprox.utils.config import AlignerV2Config
from usrembeds.models.model import AlignerV2


class UsrEmb(AlignerV2):
    """
    
    """

    def __init__(
        self,
        config: AlignerV2Config = AlignerV2Config,
        device: str = "cuda:0",
    ):
        model_state, setup, _ = AlignerV2.load_model(config.abs_file_path)
        
        config = {**config.__dict__, **setup}

        super().__init__(
            n_users=1,
            emb_size=config["emb_size"],
            prj_size=config["prj_size"],
            prj_type=config["prj"],
            aggragation=config["aggr"],
            lt=config["learnable_temp"],
            temp=config["temp"],
            drop=config["drop"],
        )
        self.device = device

        users_embedding = model_state["users.weight"]

        # remove the user embedding from the model state
        del model_state["users.weight"]

        self.load_state_dict(model_state, strict=False)
        
        self.__set_default_embedding(users_embedding) # which is an avg of the weights of the pre-trained model embeddings

        self.to(device)

    def forward(self, batch):
        index = torch.LongTensor([0 for _ in range(batch.shape[0])]).to(self.device)
        super().to(self.device)
        
        user_embedding, embeddings, _ = super().forward(index, batch)

        return user_embedding, embeddings


    def __set_default_embedding(self, users_embedding: dict) -> None:
        """
        This method is used to set the default user embedding.

        The default user embedding is set to the average of the weights of the pre-trained model.

        Args:
            config (AlignerV2Config): The configuration of the model.
        """
        average_user_embedding = users_embedding.mean(dim=0)

        self.set_user_embedding(average_user_embedding.unsqueeze(0))

    def set_user_embedding(self, user_embedding: torch.Tensor) -> None:
        """
        This method is used to set the user embedding of the model.

        Warning:
            this loses the gradient information of the user embedding.

        Args:
            user_embedding (torch.Tensor): The user embedding to set.
        """
        self.users.weight = nn.Parameter(user_embedding)

    def get_user_embedding(self) -> torch.Tensor:
        """
        This method is used to get the user embedding of the model.

        Returns:
            torch.Tensor: The user embedding.
        """
        return self.users.weight.clone()

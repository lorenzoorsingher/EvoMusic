import torch
import torch.nn as nn
from usrapprox.usrapprox.models.probabilistic import probabilistic_model_torch
from usrapprox.usrapprox.utils.config import AlignerV2Config
from usrembeds.models.model import AlignerV2


class AlignerV2Wrapper(AlignerV2):
    """
    This is a wrapper for the AlignerV2 model.

    This model will be set to **eval only** and will be used to make predictions.

    The idea here is that this will be used only to train `UsrEmb` just to verify if it works.

    This wrapper is made to run with the code of commit: 78a276fe180072119d9662cff5391ba63c8a73ee
    and the model: run_20241227_151619_best.pt
    """

    def __init__(
        self, config: AlignerV2Config = AlignerV2Config, device: str = "cuda:0"
    ):
        model_state, setup, _ = AlignerV2.load_model(config.abs_file_path)

        config = {**config.__dict__, **setup}

        super().__init__(
            n_users=config["nusers"],
            emb_size=config["emb_size"],
            prj_size=config["prj_size"],
            prj_type=config["prj"],
            aggragation=config["aggr"],
            lt=config["learnable_temp"],
            temp=config["temp"],
            drop=config["drop"],
        )

        self.to(device)
        
        self.load_state_dict(model_state)
        self.eval()

    def forward(self, idx, music_embs) -> list[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        This method is used to make predictions.

        Returns:
        - user_embedding: The user embedding.
        - embeddings: The embeddings.
        - temperature: The temperature.
        - music_score: The score of the music.        
        """
        
        if self.training:
            raise ValueError(
                "The model is in training mode but it shouldn't by design!"
            )

        with torch.no_grad():
            user_embedding, embeddings, temperature = super().forward(idx, music_embs)

            # out = user_embedding.unsqueeze(1)

            # posemb_out = embeddings[:, 0, :].unsqueeze(dim=1)
            # negemb_out = embeddings[:, 1:, :]

            # out = user_embedding.unsqueeze(1)

            # cos = nn.CosineSimilarity(dim=2, eps=1e-6)

            # possim = cos(out, posemb_out)  # .squeeze(1).cpu().detach()
            # possim = (possim + 1) / 2
            # pos_feedback_wrt_song = probabilistic_model_torch(possim)

            # out = out.repeat(1, negemb_out.shape[1], 1)
            # negsim = cos(out, negemb_out)  # .cpu().detach()

            # negsim = negsim.view(-1, negemb_out.shape[1])
            # negflat = negsim
            # # negflat = negsim.flatten()
            # negflat = (negflat + 1) / 2
            # neg_feedback_wrt_song = probabilistic_model_torch(negflat)

            # # --------------------------------------------

            # music_score = torch.cat((pos_feedback_wrt_song, neg_feedback_wrt_song), dim=1)
            
            
            # music_score_expanded = music_score.unsqueeze(-1).unsqueeze(-1)  # Shape: [16, 21, 1, 1]
            # music_score_expanded = music_score_expanded.expand(-1, -1, 13, 1)  # Shape: [16, 21, 13, 768]
            music_score = self.calculate_score(user_embedding, embeddings)


            """
            Per fare training e evaluation servono un po' di cose:
            - user_embedding e embeddings
                - per calcolare lo score $R in [0,1]$
                - per l'evaluation
            - ricordo che la loss e' fatta con contrastive learning
                - quindi user_embedding e embeddings non servono per la loss
            """

            return user_embedding, embeddings, temperature, music_score

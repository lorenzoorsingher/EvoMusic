import torch
import torch.nn as nn
from usrapprox.usrapprox.models.probabilistic import probabilistic_model_torch
from usrapprox.usrapprox.utils.config import AlignerV2Config
from usrembeds.models.model import AlignerV2


class AlignerWrapper(AlignerV2):
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

        EMB_SIZE = config["emb_size"]
        TEMP = config["temp"]
        LT = config["learnable_temp"]
        PRJ = config["prj"]
        AGGR = config["aggr"]
        DROP = config["drop"]
        MUSIC_EMB_SIZE = config["prj_size"]
        NUSERS = config["nusers"]

        super().__init__(
            n_users=NUSERS,
            emb_size=EMB_SIZE,
            prj_size=MUSIC_EMB_SIZE,
            prj_type=PRJ,
            aggragation=AGGR,
            lt=LT,
            temp=TEMP,
            drop=DROP,
        )

        self.to(device)

        self.load_state_dict(model_state)
        self.eval()

    def forward(self, idx, music_embs):
        if self.training:
            raise ValueError(
                "The model is in training mode but it shouldn't by design!"
            )

        with torch.no_grad():
            user_embedding, embeddings, _ = super().forward(idx, music_embs)

            print(f"Index shape: {idx.shape}")
            print(f"Music embeddings shape: {music_embs.shape}")

            print(f"User embedding shape: {user_embedding.shape}")
            print(f"Embeddings shape: {embeddings.shape}")

            out = user_embedding.unsqueeze(1)

            posemb_out = embeddings[:, 0, :].unsqueeze(dim=1)
            negemb_out = embeddings[:, 1:, :]

            # breakpoint()
            out = user_embedding.unsqueeze(1)

            # breakpoint()
            cos = nn.CosineSimilarity(dim=2, eps=1e-6)

            possim = cos(out, posemb_out).squeeze(1).cpu().detach()
            possim = (possim+1)/2
            pos_feedback_wrt_song = probabilistic_model_torch(possim)


            out = out.repeat(1, negemb_out.shape[1], 1)
            negsim = cos(out, negemb_out).cpu().detach()

            negsim = negsim.view(-1, negemb_out.shape[1])
            negflat = negsim.flatten()
            negflat = (negflat+1)/2
            neg_feedback_wrt_song = probabilistic_model_torch(negflat)

            return user_embedding, pos_feedback_wrt_song, neg_feedback_wrt_song

import torch

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
        self,
        config: AlignerV2Config = AlignerV2Config, device: str = "cuda:0"
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
        usr_x, music_x, _ = super().forward(idx, music_embs)

        # TODO: Get the score with the cosine similarity, normalize it between 0 and 1.

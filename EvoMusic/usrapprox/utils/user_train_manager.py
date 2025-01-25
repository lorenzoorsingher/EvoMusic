import torch

import wandb
from tqdm import tqdm

from EvoMusic.configuration import AlignerV2Config, TrainConfig, UserConfig
from EvoMusic.usrapprox.utils.user import RealUser, SynthUser, User
from EvoMusic.usrapprox.utils.user_manager import UserManager
from EvoMusic.usrapprox.utils.utils import ScoreToFeedback


class UsersTrainManager:
    def __init__(
        self,
        users: list[RealUser, SynthUser],
        users_config: UserConfig,
        train_config: TrainConfig,
        aligner_config: AlignerV2Config = AlignerV2Config(),
        device: str = "cuda",
    ):
        self.device = device

        self._score_to_feedback = ScoreToFeedback(self.device)

        self._user_manager = UserManager(
            users=users,
            users_config=users_config,
            user_delta=aligner_config.nusers,
            aligner_config=aligner_config,
            device=device,
        )

        self._train_config = train_config
        self._optimizer = None

    def __feedback_loss(
        self,
        music_scores: torch.Tensor,
        target_feedback: torch.Tensor,
        temperature: torch.Tensor,
    ):
        """
        Computes the InfoNCE loss.

        Args:
            music_scores (Tensor): Shape (batch_size, num_songs), raw scores for each song.
            target_feedback (Tensor): Shape (batch_size, num_songs), values in {-1, 1}.
            temperature (float): Temperature scaling parameter.

        Returns:
            Tensor: Scalar loss value.
        """
        # Mask positive and negative feedback
        positive_mask = (target_feedback == 1).float()  # Shape (batch_size, num_songs)
        # negative_mask = (target_feedback == -1).float()  # Shape (batch_size, num_songs)

        # Scale scores with temperature
        scaled_scores = music_scores * torch.exp(temperature)

        # Compute numerator: sum of exponentials of positive scores
        # positive_scores = scaled_scores
        positive_exp = torch.exp(scaled_scores) * positive_mask
        positive_sum = positive_exp.sum(dim=1)  # Shape (batch_size,)

        # Compute denominator: sum of exponentials of all scores (mask ensures only valid feedback)
        all_mask = (target_feedback != 0).float()  # Mask valid scores
        all_exp = torch.exp(scaled_scores) * all_mask
        all_sum = all_exp.sum(dim=1)  # Shape (batch_size,)

        # Avoid division by zero with a small epsilon
        epsilon = 1e-6

        # Compute InfoNCE loss
        info_nce_loss = -torch.log(
            (positive_sum + epsilon) / (all_sum + epsilon)
        ).mean()

        return info_nce_loss

    def train_one_step(self, tracks: torch.Tensor, user: User):
        _, _, temperature, music_score = self._user_manager.user_step(user, tracks)

        _, _, _, target_score = self._user_manager.feedback_step(user, tracks)

        target_feedback = self._score_to_feedback.get_feedback(target_score).to(
            self.device
        )

        loss = self.__feedback_loss(music_score, target_feedback, temperature)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._user_manager.usr_emb.users.parameters(), 5)
        self._optimizer.step()

        return loss.item()

    def eval(
        self,
        val_loader: torch.utils.data.DataLoader | torch.Tensor,
        user: User,
        epoch: int,
    ):

        losses = []

        # Define empty torch tensors
        music_scores = torch.empty(0).to(self.device)
        target_scores = torch.empty(0).to(self.device)

        self._user_manager.usr_emb.eval()
        with torch.no_grad():
            if isinstance(val_loader, torch.utils.data.DataLoader):
                for i, (tracks) in enumerate(
                    tqdm(val_loader, desc="Evaluating", leave=False)
                ):
                    # tracks = torch.cat((posemb, negemb), dim=1)
                    tracks = tracks.to(self.device)

                    _, _, temperature, music_score = self._user_manager.user_step(
                        user, tracks
                    )

                    _, _, _, target_score = self._user_manager.feedback_step(
                        user, tracks
                    )

                    # Calculate loss
                    target_feedback = self._score_to_feedback.get_feedback(target_score)
                    loss = self.__feedback_loss(
                        music_score, target_feedback, temperature
                    )
                    losses.append(loss.item())

                    music_scores = torch.cat((music_scores, music_score))
                    target_scores = torch.cat((target_scores, target_score))
            elif isinstance(val_loader, torch.Tensor):
                tracks = val_loader.to(self.device)

                _, _, temperature, music_score = self._user_manager.user_step(
                    user, tracks
                )

                _, _, _, target_score = self._user_manager.feedback_step(user, tracks)

                # Calculate loss
                target_feedback = self._score_to_feedback.get_feedback(target_score)
                loss = self.__feedback_loss(music_score, target_feedback, temperature)
                losses.append(loss.item())

                music_scores = torch.cat((music_scores, music_score))
                target_scores = torch.cat((target_scores, target_score))

        # Access the weights of the user embeddings
        usr_emb_weight = self._user_manager.usr_emb.users.weight[user.user_id]
        aligner_weight = self._user_manager.usr_emb.users.weight[
            user.model_reference_id
        ]

        # Compute ABS and MSE between user and aligner weights
        abs_diff = torch.abs(usr_emb_weight - aligner_weight).mean().item()
        mse_diff = torch.nn.functional.mse_loss(usr_emb_weight, aligner_weight).item()

        # Compute cosine similarity between user and aligner weights
        cosine_sim = torch.nn.functional.cosine_similarity(
            usr_emb_weight, aligner_weight, dim=-1
        )

        average_cosine_similarity_on_model = cosine_sim.mean().item()

        # Compute the mean loss
        losses = torch.tensor(losses).mean().item()

        # Compute the cosine similarity between the scores
        cosine_scores = torch.nn.functional.cosine_similarity(
            torch.Tensor(music_scores), torch.Tensor(target_scores)
        ).mean()

        # Log
        wandb.log({f"user: {user.uuid} Validation/Abs Embedding": abs_diff}, step=epoch)
        wandb.log({f"user: {user.uuid} Validation/MSE Embedding": mse_diff}, step=epoch)
        wandb.log({f"user: {user.uuid} Validation/Loss": losses}, step=epoch)
        wandb.log({
            f"user: {user.uuid} Validation/Cosine Model": average_cosine_similarity_on_model}, step=epoch
        )
        wandb.log({f"user: {user.uuid} Validation/Cosine Scores": cosine_scores}, step=epoch)

    def set_optimizer(self):
        if self._optimizer is None:
            self._optimizer = torch.optim.AdamW(
                self._user_manager.usr_emb.users.parameters(), lr=self._train_config.lr
            )

    def get_user(self, user_id: int):
        return self._user_manager[user_id]

    def finetune(self, user: User, batch: torch.Tensor, epoch: int, eval: bool = True):
        """
        Batch is expected to be a tensor on `self.device`.
        The memory is offloaded from the gpu to the cpu after each step of finetuning.

        You can iterate on the same batch multiple times, to do so just set the `epochs` parameter in the `TrainConfig` object to something greater than 1.
        """
        """
        - mettere i dati in memoria, non svuotarla MAI
        - fare il train coi dati che hai, si itera per un numero di epoche n
        """
        self.set_optimizer()

        losses = []
        for _ in tqdm(
            range(self._train_config.epochs),
            desc=f"Finetuning user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}",
        ):
            loss = self.train_one_step(batch, user)
            losses.append(loss)

        if eval:
            self.eval(batch, user, epoch)

        wandb.log({
            "Loss/finetune_user": torch.tensor(losses).mean().item()}, step=epoch
        )

    def get_user_score(self, user: User, batch: torch.Tensor):
        """
        NOT USED AS INTERNAL API OF THE CLASS.
        Get the score of a user given a batch of tracks.

        Args:
            user (User): The user for which to get the score.
            batch (torch.Tensor): The batch of tracks for which to get the score.

        Returns:
            user_embedding, embeddings, temperature, music_score
        """

        user_embedding, embeddings, temperature, music_score = (
            self._user_manager.user_step(user, batch)
        )

        return user_embedding, embeddings, temperature, music_score

    def get_reference_score(self, user: User, batch: torch.Tensor):
        """
        NOT USED AS INTERNAL API OF THE CLASS.
        Get the score of a reference user given a batch of tracks.

        Args:
            user (User): The user for which to get the score.
            batch (torch.Tensor): The batch of tracks for which to get the score.

        Returns:
            user_embedding, embeddings, temperature, music_score
        """

        user_embedding, embeddings, temperature, music_score = (
            self._user_manager.feedback_step(user, batch)
        )

        return user_embedding, embeddings, temperature, music_score

    def get_user_feedback(self, user: User, batch: torch.Tensor):
        """
        NOT USED AS INTERNAL API OF THE CLASS.
        Get the feedback of a user given a batch of tracks.

        Args:
            user (User): The user for which to get the feedback.
            batch (torch.Tensor): The batch of tracks for which to get the feedback.

        Returns:
            user_embedding, embeddings, temperature, music_score, feedback
        """

        user_embedding, embeddings, temperature, music_score = (
            self._user_manager.user_step(user, batch)
        )

        feedback = self._score_to_feedback.get_feedback(music_score).to(self.device)

        return user_embedding, embeddings, temperature, music_score, feedback

    def get_reference_feedback(self, user: User, batch: torch.Tensor):
        """
        NOT USED AS INTERNAL API OF THE CLASS.
        Get the feedback of a reference user given a batch of tracks.

        Args:
            user (User): The user for which to get the feedback.
            batch (torch.Tensor): The batch of tracks for which to get the feedback.

        Returns:
            user_embedding, embeddings, temperature, music_score, feedback
        """

        user_embedding, embeddings, temperature, music_score = (
            self._user_manager.feedback_step(user, batch)
        )

        feedback = self._score_to_feedback.get_feedback(music_score).to(self.device)

        return user_embedding, embeddings, temperature, music_score, feedback

    def clear_memory(self, user: User):
        """
        External API to clear the memory of a user.
        """
        self._user_manager.clear_memory(user)

    def save_weights():
        raise NotImplementedError("Not implemented yet.")

    def load_weights():
        raise NotImplementedError("Not implemented yet.")

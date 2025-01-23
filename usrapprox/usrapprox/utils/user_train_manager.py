import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from usrapprox.usrapprox.utils.config import AlignerV2Config, TrainConfig, UserConfig
from usrapprox.usrapprox.utils.user import RealUser, SynthUser, User
from usrapprox.usrapprox.utils.user_manager import UserManager
from usrapprox.usrapprox.utils.utils import ScoreToFeedback


class UsersTrainManager:
    def __init__(
        self,
        users: list[RealUser, SynthUser],
        users_config: UserConfig,
        writer: SummaryWriter,
        aligner_config: AlignerV2Config = AlignerV2Config(),
        device: str = "cuda",
    ):
        self.writer = writer
        self.device = device

        self._score_to_feedback = ScoreToFeedback(self.device)

        self._user_manager = UserManager(
            users=users,
            users_config=users_config,
            user_delta=aligner_config.nusers,
            aligner_config=aligner_config,
            device=device,
        )

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

    counter = 0

    def __train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        user: User,
        epoch: int,
    ):
        losses = []

        self._user_manager.clear_memory(user)

        self._user_manager.usr_emb.eval()
        for tracks in tqdm(train_loader, desc="Training", leave=False):
            # tracks = torch.cat((posemb, negemb), dim=1)
            # tracks = tracks.to(self.device)
            # print(tracks.shape)

            self._user_manager.update_memory(user, tracks)
            tracks = self._user_manager.get_memory(user)
            tracks = tracks.to(self.device)

            _, _, temperature, music_score = self._user_manager.user_step(user, tracks)

            # ids_aligner = torch.LongTensor([user.user_id] * tracks.shape[0]).to(
            #     self.device
            # )
            _, _, _, target_score = self._user_manager.feedback_step(user, tracks)

            target_feedback = self._score_to_feedback.get_feedback(target_score).to(
                self.device
            )

            loss = self.__feedback_loss(music_score, target_feedback, temperature)

            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self._user_manager.usr_emb.users.parameters(), 5
            )
            optimizer.step()

        self.writer.add_scalar(
            "Loss/Training", torch.tensor(losses).mean().item(), epoch
        )

    def __eval(self, val_loader: torch.utils.data.DataLoader, user: User, epoch: int):

        losses = []

        # Define empty torch tensors
        music_scores = torch.empty(0).to(self.device)
        target_scores = torch.empty(0).to(self.device)

        self._user_manager.usr_emb.eval()
        with torch.no_grad():
            for i, (tracks) in enumerate(
                tqdm(val_loader, desc="Evaluating", leave=False)
            ):
                # tracks = torch.cat((posemb, negemb), dim=1)
                tracks = tracks.to(self.device)

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

        # Expand aligner_weight to match the dimensions of usr_emb_weight
        # aligner_weight = aligner_weight.unsqueeze(0)  # Shape: [1, EMB_SIZE]

        # Compute cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            usr_emb_weight, aligner_weight, dim=-1
        )

        average_cosine_similarity_on_model = cosine_sim.mean().item()

        losses = torch.tensor(losses).mean().item()

        mse = torch.nn.functional.cosine_similarity(
            torch.Tensor(music_scores), torch.Tensor(target_scores)
        ).mean()

        self.writer.add_scalar("Loss/Validation", losses, epoch)
        self.writer.add_scalar(
            "Validation/Cosine Model", average_cosine_similarity_on_model, epoch
        )
        self.writer.add_scalar("Validation/Cosine Scores", mse, epoch)

    def get_user(self, user_id: int):
        return self._user_manager[user_id]

    def test_train(
        self,
        user: User,
        train_config: TrainConfig,
    ):
        train_dataloader, test_dataloader = self._user_manager.load_datasets(
            user, train_config
        )

        optimizer = torch.optim.AdamW(
            self._user_manager.usr_emb.users.parameters(), lr=train_config.lr
        )

        for epoch in tqdm(
            range(train_config.epochs),
            desc=f"Training user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}",
        ):
            self.__train(train_dataloader, optimizer, user, epoch)

            self.__eval(test_dataloader, user, epoch)

    def finetune():
        raise NotImplementedError("Not implemented yet.")

    def save_weights():
        raise NotImplementedError("Not implemented yet.")

    def load_weights():
        raise NotImplementedError("Not implemented yet.")

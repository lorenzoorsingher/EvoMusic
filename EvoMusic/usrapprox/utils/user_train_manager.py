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

        # wandb.init(
        #     project="EvoUsers",
        #     name="UsersTrainManager",
        #     config={
        #         "users_config": users_config.__dict__,
        #         "train_config": train_config.__dict__,
        #         "aligner_config": aligner_config.__dict__,
        #         "device": device,
        #     }
        # )

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

    def train_one_step(
        self, tracks: torch.Tensor, target_feedback: torch.Tensor, user: User
    ):
        _, _, temperature, music_score = self._user_manager.user_step(user, tracks)

        loss = self.__feedback_loss(music_score, target_feedback, temperature)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._user_manager.usr_emb.users.parameters(), 5)
        self._optimizer.step()

        return loss.item()

    def eval_finetune(self, user: User, epoch):
        if user.test_dataloader is None:
            self._user_manager.load_dataset(user, self._train_config, "test")

        dataloader = self.get_test_dataloader(user)
        self.eval(dataloader, user, epoch)

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
        wandb.log({f"user: {user.uuid} Validation/Abs Embedding": abs_diff})
        wandb.log({f"user: {user.uuid} Validation/MSE Embedding": mse_diff})
        wandb.log({f"user: {user.uuid} Validation/Loss": losses})
        wandb.log(
            {
                f"user: {user.uuid} Validation/Cosine Model": average_cosine_similarity_on_model
            }
        )
        wandb.log({f"user: {user.uuid} Validation/Cosine Scores": cosine_scores})

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

        -  You can iterate on the same batch multiple times, to do so just set the `epochs` parameter in the `TrainConfig` object to something greater than 1.
        OR
        - You can set the `minibatch` parameter in the `UserConfig` object to True to use minibatching.

        Note: The memory is offloaded to the cpu after each step of finetuning.
        """
        self.set_optimizer()
        if epoch > 0:
            wandb.log({"Epoch": epoch - 1})

        losses = []

        # Update memory
        self.update_memory(user, batch)
        self.set_memory_device(user, self.device)
        data, target_feedback = self.get_memory(user)

        if user.minibatch:
            # USE MINIBATCHING
            minibatches = self.shuffle_and_create_minibatches(
                data, target_feedback, user._memory._stored_elements
            )

            for data, target_feedback in tqdm(
                minibatches,
                desc=f"Finetuning user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}", leave=False,
            ):
                loss = self.train_one_step(data, target_feedback, user)
                losses.append(loss)
        else:
            for _ in tqdm(
                range(self._train_config.epochs),
                desc=f"Finetuning user {user.uuid} | {user.user_id} | ref: {user.model_reference_id}", leave=False,
            ):
                loss = self.train_one_step(data, target_feedback, user)
                losses.append(loss)

        if eval:
            if isinstance(user, SynthUser):
                self.eval_finetune(user, epoch)
            else:
                # not implemented error
                raise NotImplementedError(
                    "Not implemented! Eval for RealUser is a pain."
                )

        # Offload memory to cpu
        self.set_memory_device(user, torch.device("cpu"))

        wandb.log({"epoch": epoch})
        wandb.log({"Loss/finetune_user": torch.tensor(losses).mean().item()})

    def shuffle_and_create_minibatches(self, memory, feedback, batch_size):
        """
        Shuffle data along the song dimension and create minibatches.
        The memory and feedback tensors are assumed to already be aligned in their final concatenated shapes.

        Args:
            memory (torch.Tensor): The memory tensor with shape [1, total_samples, n_song_embedding, ...].
            feedback (torch.Tensor): The feedback tensor with shape [1, total_samples, n_song_feedbacks].
            batch_size (int): The size of each minibatch (number of memory elements per minibatch).

        Returns:
            list of tuples: A list where each element is a tuple (memory_batch, feedback_batch).
        """
        if memory.shape[1] != feedback.shape[1]:
            raise ValueError(
                "Memory and feedback must have the same number of samples along the first dimension."
            )

        # Extract key dimensions
        total_samples = memory.shape[1]
        # n_song_embedding = memory.shape[2]
        # song_dim = memory.shape[3]
        # n_feedbacks = feedback.shape[2]

        # Ensure the total samples can be evenly divided by batch_size
        if total_samples % batch_size != 0:
            raise ValueError("Total samples must be divisible by the batch size.")

        # Shuffle indices for the total_samples dimension
        indices = torch.randperm(total_samples)

        # Shuffle both tensors along the relevant dimension
        shuffled_memory = memory[:, indices, :, :]
        shuffled_feedback = feedback[:, indices]

        # Split the shuffled tensors into minibatches
        minibatches = []
        for i in range(batch_size):
            start_idx = i * (total_samples // batch_size)
            end_idx = start_idx + (total_samples // batch_size)

            # Extract minibatches
            memory_batch = shuffled_memory[:, start_idx:end_idx, :, :]
            feedback_batch = shuffled_feedback[:, start_idx:end_idx]

            minibatches.append((memory_batch, feedback_batch))

        return minibatches


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

    def update_memory(self, user: User, batch: torch.Tensor):
        """
        External API to set the memory of a user.
        It calculates/gets the feedback from a real/synthetic user and updates the memory of the user.
        """

        _, _, _, target_score = self._user_manager.feedback_step(user, batch)

        target_feedback = self._score_to_feedback.get_feedback(target_score).to(
            self.device
        )

        # print(f"target_feedback: {target_feedback.shape}")
        # print(f"batch: {batch.shape}")

        self._user_manager.update_memory(user, batch, target_feedback)

    def get_memory(self, user: User) -> list[torch.Tensor, torch.Tensor]:
        """
        External API to get the memory of a user.
        """
        return self._user_manager.get_memory(user)

    def get_train_dataloader(self, user: User):
        """
        External API to get the train dataloader of a user.
        """
        return user.train_dataloader

    def get_test_dataloader(self, user: User):
        """
        External API to get the test dataloader of a user.
        """
        return user.test_dataloader

    def set_memory_device(self, user: User, device: torch.device):
        """
        External API to set the device of the memory of a user.
        """
        self._user_manager.set_memory_device(user, device)

    def save_weights():
        raise NotImplementedError("Not implemented yet.")

    def load_weights():
        raise NotImplementedError("Not implemented yet.")

import torch

class MemoryBuffer:
    def __init__(self, memory_length: int):
        """
        Initialize the memory buffer.

        Args:
            memory_length (int): The number of elements to store in memory.
        """
        self._memory_length = memory_length
        self._memory = None
        self._feedback_memory = None
        self._current_index = 0
        self._stored_elements = 0  # Track the number of elements currently stored

    def add_to_memory(self, batch: torch.Tensor, feedback: torch.Tensor):
        """
        Add a batch tensor and its feedback to the memory. If memory is not initialized, it is created based on the first batch.

        Args:
            batch (torch.Tensor): The tensor to add to memory.
            feedback (torch.Tensor): The feedback tensor to add to memory.
        """
        batch = batch.squeeze(0)  # Remove the first dimension (batch size = 1)
        feedback = feedback.squeeze(0)  # Remove the first dimension

        if self._memory is None:
            sequence_length, *feature_dims = batch.shape
            feedback_length = feedback.shape[0]

            if feedback_length != sequence_length:
                raise ValueError("Feedback length must match the sequence length of the batch.")

            self._memory = torch.empty(
                (self._memory_length * sequence_length, *feature_dims),
                dtype=batch.dtype,
                device=batch.device,
            )

            self._feedback_memory = torch.empty(
                (self._memory_length * sequence_length,),
                dtype=feedback.dtype,
                device=feedback.device,
            )

        start_index = self._current_index * batch.shape[0]
        end_index = start_index + batch.shape[0]

        self._memory[start_index:end_index] = batch
        self._feedback_memory[start_index:end_index] = feedback

        self._current_index = (self._current_index + 1) % self._memory_length

        if self._stored_elements < self._memory_length:
            self._stored_elements += 1

    def empty_memory(self):
        """
        Empty the memory buffer by resetting the index and stored element count.
        """
        self._current_index = 0
        self._stored_elements = 0

    def to(self, device: torch.device):
        """
        Move the memory to a new device.

        Args:
            device (torch.device): The device to move the memory to.
        """
        if self._memory is not None:
            self._memory = self._memory.to(device)
        if self._feedback_memory is not None:
            self._feedback_memory = self._feedback_memory.to(device)

    @property
    def memory(self):
        """
        Get the memory and feedback as tensors.

        Returns:
            tuple: A tuple containing two tensors:
                - A tensor with all stored batches, flattened along the first dimension.
                - A tensor with all stored feedback, flattened along the first dimension.
        """
        if self._memory is None or self._stored_elements == 0:
            return torch.empty(0), torch.empty(0)  # Return empty tensors if no memory has been added

        stored_length = self._stored_elements * (self._memory.shape[0] // self._memory_length)

        if self._stored_elements < self._memory_length:
            data = self._memory[:stored_length]
            feedback_data = self._feedback_memory[:stored_length]
        else:
            start_index = self._current_index * (self._memory.shape[0] // self._memory_length)
            data = torch.cat((
                self._memory[start_index:],
                self._memory[:start_index],
            ), dim=0)
            feedback_data = torch.cat((
                self._feedback_memory[start_index:],
                self._feedback_memory[:start_index],
            ), dim=0)

        # Reshape for output
        data = data.view(1, -1, *data.shape[1:])
        feedback_data = feedback_data.view(1, -1)
        return data, feedback_data

def shuffle_and_create_minibatches(memory, feedback, batch_size):
    """
    Shuffle data along the batch_size * n_elements dimension and create minibatches.
    The memory and feedback tensors are assumed to already be aligned in their final concatenated shapes.

    Args:
        memory (torch.Tensor): The memory tensor with shape [1, total_samples, n_song_embedding, ...].
        feedback (torch.Tensor): The feedback tensor with shape [1, total_samples, n_song_feedbacks].
        batch_size (int): The size of each minibatch.

    Returns:
        list of tuples: A list where each element is a tuple (memory_batch, feedback_batch).
    """
    if memory.shape[1] != feedback.shape[1]:
        raise ValueError(
            "Memory and feedback must have the same number of samples along the first dimension."
        )

    # Remove the first dimension (batch size = 1)
    memory = memory.squeeze(0)
    feedback = feedback.squeeze(0)

    # Determine the number of elements per batch
    n_elements_per_batch = memory.shape[0] // batch_size

    # Shuffle indices for the batch_size * n_elements dimension
    indices = torch.randperm(memory.shape[0])

    # Shuffle both tensors along the relevant dimension
    shuffled_memory = memory[indices]
    shuffled_feedback = feedback[indices]

    # Reshape shuffled data back into minibatches
    memory_batches = shuffled_memory.view(batch_size, n_elements_per_batch, *memory.shape[1:])
    feedback_batches = shuffled_feedback.view(batch_size, n_elements_per_batch)

    # Combine memory and feedback minibatches
    minibatches = [(m.unsqueeze(0), f.unsqueeze(0)) for m, f in zip(memory_batches, feedback_batches)]

    return minibatches

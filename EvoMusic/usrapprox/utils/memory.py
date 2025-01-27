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
        if self._memory is None:
            batch_shape = batch.shape
            feedback_shape = feedback.shape

            if feedback_shape[0] != batch_shape[0]:
                raise ValueError("Feedback batch size must match input batch size.")

            self._memory = torch.empty(
                (self._memory_length,) + batch_shape,
                dtype=batch.dtype,
                device=batch.device,
            )

            self._feedback_memory = torch.empty(
                (self._memory_length,) + feedback_shape[1:],
                dtype=feedback.dtype,
                device=feedback.device,
            )

        if batch.shape != self._memory.shape[1:] or feedback.shape[1:] != self._feedback_memory.shape[1:]:
            raise ValueError("Batch or feedback shape does not match memory shape.")

        self._memory[self._current_index] = batch
        self._feedback_memory[self._current_index] = feedback
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

        if self._stored_elements < self._memory_length:
            data = self._memory[: self._stored_elements]
            feedback_data = self._feedback_memory[: self._stored_elements]
        else:
            data = torch.cat(
                (
                    self._memory[self._current_index :],
                    self._memory[: self._current_index],
                ),
                dim=0,
            )
            feedback_data = torch.cat(
                (
                    self._feedback_memory[self._current_index :],
                    self._feedback_memory[: self._current_index],
                ),
                dim=0,
            )

        return data.view(-1, *data.shape[2:]), feedback_data.view(-1, *feedback_data.shape[1:])

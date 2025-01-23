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
        self._current_index = 0
        self._stored_elements = 0  # Track the number of elements currently stored

    def add_to_memory(self, batch: torch.Tensor):
        """
        Add a batch tensor to the memory. If memory is not initialized, it is created based on the first batch.

        Args:
            batch (torch.Tensor): The tensor to add to memory.
        """
        if self._memory is None:
            batch_shape = batch.shape
            self._memory = torch.empty((self._memory_length,) + batch_shape, dtype=batch.dtype, device=batch.device)

        if batch.shape != self._memory.shape[1:]:
            raise ValueError("Batch shape does not match memory batch shape.")

        self._memory[self._current_index] = batch
        self._current_index = (self._current_index + 1) % self._memory_length
        
        if self._stored_elements < self._memory_length:
            self._stored_elements += 1

    def empty_memory(self):
        """
        Empty the memory buffer by resetting the index and stored element count.
        """
        self._current_index = 0
        self._stored_elements = 0

    @property
    def memory(self) -> torch.Tensor:
        """
        Get the memory as a single tensor.

        Returns:
            torch.Tensor: A tensor containing all stored batches, flattened into a single tensor along the first dimension.
        """
        if self._memory is None or self._stored_elements == 0:
            return torch.empty(0)  # Return an empty tensor if no memory has been added

        if self._stored_elements < self._memory_length:
            data = self._memory[:self._stored_elements]
        else:
            data = torch.cat((
                self._memory[self._current_index:], 
                self._memory[:self._current_index]
            ), dim=0)

        return data.view(-1, *data.shape[2:])
from abc import ABC, abstractmethod

import torch


class Dataset(ABC):
    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        ...

    @abstractmethod
    def __getitem__(self, idx) -> tuple[torch.Tensor, ...]:
        """
        Get sample(s) from the dataset at index idx.
        
        Args:
            idx: Index of the sample to retrieve.
        
        Returns:
            A tuple containing the sample data.
        """
        ...
    
    def sample_batch(self, num_samples: int):
        """
        Sample a number of random samples from the dataset.
        
        Args:
            num_samples: Number of samples to retrieve.
        
        Returns:
            A list of tuples containing the sampled data.
        """
        if num_samples > len(self):
            raise ValueError("num_samples exceeds dataset size")
        indices = torch.randint(0, len(self), (num_samples,))
        return self[indices]
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...
    
    @abstractmethod
    def to(self, device: torch.device) -> 'Dataset':
        ...

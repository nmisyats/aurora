from abc import ABC, abstractmethod
from typing import Callable, Any

import torch
import torch.nn as nn

from aurora.models import FluxModel

# Type hints for clarity
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
RegularizationFunction = Callable[[torch.Tensor], torch.Tensor]

# Simple configuration for each loss term
class LossTerm(nn.Module, ABC):
    """Configuration for a single loss term"""
    def __init__(self,
                 model: FluxModel,
                 weight: float = 1.0,
                 batch_size: int = 1000,
                 name: str = None):
        super().__init__()
        self.model = model
        self.weight = weight
        self.batch_size = batch_size
        self.name = name if name is not None else self.__class__.__name__.lower()
        self.history = []
    
    @property
    def device(self):
        return self.model.device
    
    def forward(self, batch_in: Any) -> torch.Tensor:
        loss = self.eval_raw_loss(batch_in)
        weighted_loss = self.weight * loss
        self.history.append(weighted_loss.item())
        return weighted_loss
    
    @abstractmethod
    def sample_batch(self) -> Any:
        ...
    
    @abstractmethod
    def eval_raw_loss(self, batch_in: Any) -> torch.Tensor:
        ...


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class ElectronFluxModel(ABC):
    @abstractmethod
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        ...

class TrainableFluxModel(ElectronFluxModel, nn.Module):
    def flux(self, xy: torch.Tensor):
        orig_shape = xy.shape[:-1] # (k1, k2, ..., kn, 2)
        xy = xy.reshape(-1, 2) # (N, 2)
        f = self.forward(xy) # (N, n_bins)
        n_bins = f.shape[-1]
        return f.reshape(*orig_shape, n_bins)

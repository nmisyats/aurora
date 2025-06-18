from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn

from aurora.reconstruction import Reconstruction
from aurora.dataset import CameraRaysDataset, RadarPointsDataset

LossFn = Callable[[Reconstruction], torch.Tensor]

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, recon: Reconstruction):
        ...

class CombinedLoss(Loss):
    """Manager class for handling multiple loss functions"""
    
    def __init__(self, losses: dict[str, Loss], weights: dict[str, float]):
        self.loss_functions = losses
        self.loss_weights = weights
        self.loss_history: dict[str, list[float]] = {}
    
    def forward(self, recon: Reconstruction):
        total_loss = torch.tensor(0.0, device=recon.device)
        
        for name, loss_fn in self.loss_functions.items():
            if self.loss_weights.get(name, 0.0) == 0.0:
                continue
            
            loss_value = loss_fn(recon)
            weighted_loss = self.loss_weights[name] * loss_value
            
            total_loss = total_loss + weighted_loss
            self.loss_history[name].append(loss_value.item())
        
        self.loss_history["total"].append(total_loss.item())
        return total_loss
    
    @property
    def last_loss(self):
        return {name: values[-1] for name, values in self.loss_history.items()}

class RayLoss(Loss):
    def __init__(
            self,
            batch_size: int,
            ray_data: CameraRaysDataset,
            ray_bins: int
        ):
        super().__init__()
        self.batch_size = batch_size
        self.ray_data = ray_data
        self.ray_bins = ray_bins
    
    def forward(self, recon: Reconstruction):
        ro, rd, tn, tf, g_ref = self.ray_data.random_sample(self.batch_size)
        g_pred = recon.int_emis_ray(ro, rd, tn, tf, self.ray_bins)
        loss = (g_pred - g_ref)**2
        # loss = loss / loss.sum()
        return loss.mean()

class RadarLoss(Loss):
    def __init__(
            self,
            batch_size: int,
            radar_data: RadarPointsDataset
        ):
        super().__init__()
        self.batch_size = batch_size
        self.radar_data = radar_data
    
    def forward(self, recon: Reconstruction):
        p, d_ref = self.radar_data.random_sample(self.batch_size)
        d_pred = recon.elec_dens(p)
        loss = (d_pred - d_ref)**2
        # loss = loss / loss.sum()
        return loss.mean()

class FluxSmoothnessLoss(Loss):
    def __init__(self):
        super().__init__()
        raise NotImplementedError
    
    def forward(self, recon: Reconstruction):
        raise NotImplementedError

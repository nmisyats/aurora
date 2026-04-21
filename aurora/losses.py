import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel
from aurora.datasets import RayBatch, RadarBatch
from aurora.samplers import RaySampler

__all__ = [
    "ray_loss",
    "radar_loss",
    "spectral_smoothness_loss",
    "RayLoss",
    "RadarLoss",
    "SpectralSmoothnessLoss"
]


def ray_loss(model: FluxModel, batch: RayBatch, sampler: RaySampler) -> torch.Tensor:
    ro, rd, tn, tf, g_target = batch
    g_pred = model.integrate_emis_along_ray(ro, rd, tn, tf, sampler)
    return F.mse_loss(g_pred, g_target)

def radar_loss(model: FluxModel, batch: RadarBatch) -> torch.Tensor:
    p, d_target = batch
    d_pred = model.get_electron_density(p)
    return F.mse_loss(d_pred, d_target)

def spectral_smoothness_loss(model: FluxModel, xy: torch.Tensor) -> torch.Tensor:
    out = model.forward(xy)
    log_f_pred = out["log_f"]
    # Calculate bin centers and widths
    energy_bins = model.energy_bins
    bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])  # (num_bins,)
    # bin_widths = energy_bins[1:] - energy_bins[:-1]  # (num_bins,)
    # Log-scale bin centers for proper weighting
    log_centers = torch.log(bin_centers)
    log_spacing = torch.diff(log_centers)  # (num_bins-1,)
    # Second-order derivative approximation in log-energy space
    first_diff = torch.diff(log_f_pred, dim=1)  # (batch_size, num_bins-1)
    normalized_diff = first_diff / log_spacing.unsqueeze(0)  # (batch_size, num_bins-1)
    second_diff = torch.diff(normalized_diff, dim=1)  # (batch_size, num_bins-2)
    smoothness_penalty = torch.mean(second_diff**2)
    return smoothness_penalty

class RayLoss(nn.Module):
    def __init__(self, sampler: RaySampler):
        super().__init__()
        self.sampler = sampler

    def forward(self, model: FluxModel, batch: RayBatch):
        return ray_loss(model, batch, self.sampler)

class RadarLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, model: FluxModel, batch: RadarBatch):
        return radar_loss(model, batch)

class SpectralSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(model: FluxModel, xy: torch.Tensor):
        return spectral_smoothness_loss(model, xy)

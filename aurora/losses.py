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
]


def ray_loss(model: FluxModel, batch: RayBatch, sampler: RaySampler):
    ro, rd, tn, tf, g_target, wl = batch
    g_pred = model.integrate_emis_along_ray(ro, rd, tn, tf, sampler, wl)
    return F.mse_loss(g_pred, g_target)


def radar_loss(model: FluxModel, batch: RadarBatch):
    p, d_target = batch
    d_pred = model.get_electron_density(p)
    return F.mse_loss(d_pred, d_target)


def spectral_smoothness_loss(model: FluxModel, xy: torch.Tensor):
    # Assumings bins are logarithmically spaced
    out = model(xy)
    log_f = out["log_f"]
    d2log_f = torch.diff(log_f, n=2, dim=-1) # (B, num_bins-2)
    return torch.mean(d2log_f.square())

from functools import lru_cache

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


@lru_cache
def _dlogE(energy_bins: torch.Tensor):
    E = 0.5 * (energy_bins[:-1] + energy_bins[1:]) # (num_bins,)
    logE = torch.log(E)
    return torch.diff(logE) # (num_bins-1,)

def spectral_smoothness_loss(model: FluxModel, xy: torch.Tensor):
    dlogE = _dlogE(model.energy_bins)
    out = model(xy)
    log_f_pred = out["log_f"]
    fst_diff = torch.diff(log_f_pred, dim=1) # (B, num_bins-1)
    nml_diff = fst_diff / dlogE.unsqueeze(0)
    snd_diff = torch.diff(nml_diff, dim=1) # (B, num_bins-2)
    return torch.mean(snd_diff**2)

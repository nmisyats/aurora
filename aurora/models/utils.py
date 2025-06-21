from pathlib import Path

import torch

from aurora.models.flux_model import FluxModel
from aurora.models.grid_flux import GridSampledFlux
from aurora.data import load_config, load_3d_grid_data


def save_model(model: FluxModel, path: Path | str):
    torch.save(model, path)

def load_model(path: Path | str, device=torch.device("cpu")) -> FluxModel:
    return torch.load(path, map_location=device, weights_only=False)

def load_grid_model(
        flux_path: Path | str,
        config_path: Path | str,
        device=torch.device("cpu")
    ) -> GridSampledFlux:
    flux_path = Path(flux_path)
    config_path = Path(config_path)
    config = load_config(config_path)
    return GridSampledFlux(
        frame=config.frame,
        bbox=config.bbox,
        emis_mat=config.physics.emis_mat,
        dens_mat=config.physics.dens_mat,
        energy_bins=config.physics.energy_bins,
        altitude_bins=config.physics.altitude_bins,
        data=load_3d_grid_data(flux_path)
    ).to(device).eval()
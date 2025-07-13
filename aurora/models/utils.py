import torch

from aurora.models.flux_model import FluxModel
from aurora.models.grid_flux import GridSampledFlux
from aurora.data import PathLike, load_config, load_3d_grid_data


def save_model(model: FluxModel, path: PathLike):
    torch.save(model, path)

def load_model(path: PathLike, device=torch.device("cpu")) -> FluxModel:
    return torch.load(path, map_location=device, weights_only=False)

def load_grid_model(
        flux_path: PathLike,
        config_path: PathLike,
        device=torch.device("cpu")
    ) -> GridSampledFlux:
    config = load_config(config_path)
    data = load_3d_grid_data(flux_path)
    grid_flux = GridSampledFlux(config, data)
    return grid_flux.to(device).eval()
import torch

from aurora.models.flux_model import FluxModel
from aurora.models.grid_flux import GridSampledFlux
from aurora.data import PathLike, load_config, load_flux_data


def save_model(model: FluxModel, path: PathLike):
    torch.save(model, path)

def load_model(path: PathLike) -> FluxModel:
    return torch.load(path, weights_only=False)

def load_grid_model(flux_path: PathLike, config_path: PathLike):
    config = load_config(config_path)
    data = load_flux_data(flux_path)
    grid_f = GridSampledFlux(config, data)
    return grid_f.eval()
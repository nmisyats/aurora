from typing import Optional

import numpy as np
import torch

from .common import PathLike
from .arrays import load_matrix_data, load_3d_grid_data


def load_flux_data(
        dat_path: PathLike,
        nx: Optional[int] = None,
        ny: Optional[int] = None,
        nE: Optional[int] = None,
        mul_by_pi: bool = False
    ):
    if nx and ny and nE:
        flux = np.loadtxt(dat_path, dtype=np.float32)
        flux = np.reshape(flux, (nE, nx, ny), order="F")
        flux = flux.transpose(0, 2, 1)
        flux = np.transpose(flux, (1, 2, 0))
        flux = torch.from_numpy(flux)
    else:
        flux = load_3d_grid_data(dat_path)
    if mul_by_pi:
        flux = flux * torch.pi
    return flux


def load_emission_matrix(dat_path: PathLike):
    return load_matrix_data(dat_path).T


def load_density_matrix(dat_path: PathLike):
    return load_matrix_data(dat_path).T.square()


def load_altitude_bins(dat_path: PathLike):
    return load_matrix_data(dat_path).flatten()


def load_energy_bins(dat_path: PathLike):
    return load_matrix_data(dat_path).flatten()


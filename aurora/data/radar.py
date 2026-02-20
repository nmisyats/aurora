from dataclasses import dataclass

import numpy as np
import torch

from .common import PathLike


@dataclass
class RadarPointCloud:
    latitudes: torch.Tensor
    longitudes: torch.Tensor
    altitudes: torch.Tensor
    densities: torch.Tensor


def load_radar_point_cloud(dat_path: PathLike):
    data = np.loadtxt(dat_path, dtype=np.float32)
    data = torch.from_numpy(data)
    h, lat, lon, d = data.T
    return RadarPointCloud(lat, lon, h, d)


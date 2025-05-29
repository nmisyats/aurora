import numpy as np
import torch
from pathlib import Path
from typing import Union, Tuple, List


Coord2DLike = Union[Tuple[float, float], List[float], torch.Tensor]
Coord3DLike = Union[Tuple[float, float, float], List[float], torch.Tensor]

def bounds2d_to_tuple(min_point: Coord2DLike, max_point: Coord2DLike) -> Tuple[float, float, float, float]:
    """
    Converts ((x_min, y_min), (x_max, y_max)) to (x_min, x_max, y_min, y_max).
    """
    x_min, y_min = float(min_point[0]), float(min_point[1])
    x_max, y_max = float(max_point[0]), float(max_point[1])
    return x_min, x_max, y_min, y_max


def ranges2d_to_tuple(x_range: Coord2DLike, y_range: Coord2DLike) -> Tuple[float, float, float, float]:
    """
    Converts ((x_min, x_max), (y_min, y_max)) to (x_min, x_max, y_min, y_max).
    """
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    return x_min, x_max, y_min, y_max

def bounds3d_to_tuple(min_point: Coord3DLike, max_point: Coord3DLike) -> Tuple[float, float, float, float, float, float]:
    """
    Converts ((x_min, y_min, z_min), (x_max, y_max, z_max)) to (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    x_min, y_min, z_min = float(min_point[0]), float(min_point[1]), float(min_point[2])
    x_max, y_max, z_max = float(max_point[0]), float(max_point[1]), float(max_point[2])
    return x_min, x_max, y_min, y_max, z_min, z_max

def ranges3d_to_tuple(x_range: Coord2DLike, y_range: Coord2DLike, z_range: Coord2DLike) -> Tuple[float, float, float, float, float, float]:
    """
    Converts ((x_min, x_max), (y_min, y_max), (z_min, z_max)) to (x_min, x_max, y_min, y_max, z_min, z_max).
    """
    x_min, x_max = float(x_range[0]), float(x_range[1])
    y_min, y_max = float(y_range[0]), float(y_range[1])
    z_min, z_max = float(z_range[0]), float(z_range[1])
    return x_min, x_max, y_min, y_max, z_min, z_max

def load_matrix_data(dat_path: Path | str):
    mat = []
    with open(dat_path, "r") as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            mat.append(row)
    mat = torch.tensor(mat, dtype=torch.float32)
    return mat

def load_3d_grid_data(dat_path: Path | str):
    data = np.loadtxt(dat_path)
    indices = data[:, :3].astype(int)
    values = data[:, 3]
    # Determine array shape from max index values
    ni, nj, nk = indices.max(axis=0) + 1
    array = np.zeros((ni, nj, nk), dtype=values.dtype)
    # Assign values
    array[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return torch.from_numpy(array).to(torch.float32)

def save_3d_grid_data(array: np.ndarray, file_path: Path):
    ni, nj, nk = array.shape
    indices = np.indices((ni, nj, nk)).reshape(3, -1).T  # Generate i, j, k indices efficiently
    values = array.ravel().reshape(-1, 1)  # Flatten array values
    data = np.hstack((indices, values))  # Combine indices with values
    np.savetxt(file_path, data, fmt="%d %d %d %.6f")  # Save to file with formatting

def xy_grid(
        xy_min: torch.Tensor,
        xy_max: torch.Tensor,
        res_x: int,
        res_y: int,
        device: torch.device | None = None
    ) -> torch.Tensor:
    if device is None:
        device = xy_min.device
    x = torch.linspace(xy_min[0], xy_max[0], res_x, device=device)
    y = torch.linspace(xy_min[1], xy_max[1], res_y, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack((xx, yy), dim=-1)
    return xy

def xyz_grid(
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
        res_x: int,
        res_y: int,
        res_z: int,
        device: torch.device | None = None
    ) -> torch.Tensor:
    if device is None:
        device = xyz_min.device
    x = torch.linspace(xyz_min[0], xyz_max[0], res_x, device=device)
    y = torch.linspace(xyz_min[1], xyz_max[1], res_y, device=device)
    z = torch.linspace(xyz_min[2], xyz_max[2], res_z, device=device)
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    xyz = torch.stack((xx, yy, zz), dim=-1)
    return xyz

def downsample_image(image: torch.Tensor, factor: int) -> np.ndarray:
    """
    Downsamples a 2D or 3D image (e.g., grayscale or RGB) by picking every `factor`-th pixel.
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be >= 1")
    
    # Handle 2D (grayscale) or 3D (color) images
    if image.ndim == 2:
        return image[::factor, ::factor]
    else:
        return image[::factor, ::factor, :]

import numpy as np
from functools import wraps
import torch
from pathlib import Path
from typing import NamedTuple

class MinMax(NamedTuple):
    min: float
    max: float

def to_minmax(lst):
    if not isinstance(lst, (list, tuple)):
        raise TypeError("Value must be a list or tuple")
    if len(lst) != 2:
        raise ValueError("List must have exactly two elements")
    return MinMax(float(lst[0]), float(lst[1]))

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

def numpify(func=None, *, device="cpu"):
    if func is None:
        return lambda f: numpify(f, device=device)

    @wraps(func)
    def wrapper(*args, **kwargs):
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            return x

        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            elif isinstance(x, (list, tuple)):
                return type(x)(to_numpy(i) for i in x)
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            return x

        args_t = tuple(to_tensor(a) for a in args)
        kwargs_t = {k: to_tensor(v) for k, v in kwargs.items()}
        result = func(*args_t, **kwargs_t)
        return to_numpy(result)

    return wrapper

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
    xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2)
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
    xyz = torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)
    return xyz
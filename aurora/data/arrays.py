import numpy as np
import torch

from .common import PathLike


def load_matrix_data(dat_path: PathLike):
    mat = np.loadtxt(dat_path, dtype=np.float32)
    return torch.from_numpy(mat)


def save_matrix_data(tensor: torch.Tensor, dat_path: PathLike):
    mat = tensor.detach().cpu().numpy()
    np.savetxt(dat_path, mat, fmt="%.6f", delimiter=" ")


def load_3d_grid_data(dat_path: PathLike):
    data = np.loadtxt(dat_path)
    indices = data[:, :3].astype(int)
    values = data[:, 3]
    ni, nj, nk = indices.max(axis=0) + 1
    array = np.zeros((ni, nj, nk), dtype=values.dtype)
    array[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return torch.from_numpy(array).to(torch.float32)


def save_3d_grid_data(array: torch.Tensor, file_path: PathLike):
    array = array.numpy(force=True)
    ni, nj, nk = array.shape
    indices = np.indices((ni, nj, nk)).reshape(3, -1).T
    values = array.ravel().reshape(-1, 1)
    data = np.hstack((indices, values))
    np.savetxt(file_path, data, fmt="%d %d %d %.6f")


def load_2d_grid_data(dat_path: PathLike):
    data = np.loadtxt(dat_path)
    indices = data[:, :2].astype(int)
    values = data[:, 2]
    ni, nj = indices.max(axis=0) + 1
    array = np.zeros((ni, nj), dtype=values.dtype)
    array[indices[:, 0], indices[:, 1]] = values
    return torch.from_numpy(array).to(torch.float32)


def save_2d_grid_data(array: torch.Tensor, file_path: PathLike):
    array = array.numpy(force=True)
    ni, nj = array.shape
    indices = np.indices((ni, nj)).reshape(2, -1).T
    values = array.ravel().reshape(-1, 1)
    data = np.hstack((indices, values))
    np.savetxt(file_path, data, fmt="%d %d %.6f")

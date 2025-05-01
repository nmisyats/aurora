import numpy as np
from pathlib import Path

def save_3d_array(array: np.ndarray, file_path: Path):
    ni, nj, nk = array.shape
    indices = np.indices((ni, nj, nk)).reshape(3, -1).T  # Generate i, j, k indices efficiently
    values = array.ravel().reshape(-1, 1)  # Flatten array values
    data = np.hstack((indices, values))  # Combine indices with values
    np.savetxt(file_path, data, fmt="%d %d %d %.6f")  # Save to file with formatting

def load_3d_array(file_path: Path) -> np.ndarray:
    data = np.loadtxt(file_path)
    indices = data[:, :3].astype(int)
    values = data[:, 3]
    # Determine array shape from max index values
    ni, nj, nk = indices.max(axis=0) + 1
    array = np.zeros((ni, nj, nk), dtype=values.dtype)
    # Assign values
    array[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return array
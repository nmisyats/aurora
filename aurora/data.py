from pathlib import Path

from schema import Schema, Optional, And, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import torch
import numpy as np

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.models import ReferenceFlux

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)


# _minmax_tuple = And(Use(lambda lst: (float(lst[0]), float(lst[1])), lambda p: p[0] < p[1]))
# _float = Use(float)

# def resolve_relative_to_base(target_path: Path, base_path: Path) -> Path:
#     base_path = Path(base_path)
#     target_path = Path(target_path)
#     if target_path.is_absolute():
#         return target_path.resolve()
#     abs_base = base_path.resolve()
#     return (abs_base / target_path).resolve()

# def path_validator(base_path: Path):
#     return And(
#         Use(Path),
#         Use(lambda p: resolve_relative_to_base(p, base_path)),
#         lambda p: p.exists(),
#         error="Must be a valid path"
#     )

def load_camera_images(cam_dir: Path | str):
    cam_dir = Path(cam_dir)
    image = load_matrix_data(cam_dir / "image.dat")
    azimuth = load_matrix_data(cam_dir / "az_cam.dat")
    zenith = load_matrix_data(cam_dir / "ze_cam.dat")
    return image, azimuth, zenith

def load_cameras(cam_pos_set: Path | str, cam_images_dir: Path | str):
    cam_pos_set = Path(cam_pos_set)
    cam_images_dir = Path(cam_images_dir)
    positions = load_camera_positions(cam_pos_set)
    cameras = []
    for cam_name, cam_pos in positions.items():
        cam_dir = cam_images_dir / cam_name
        image, azimuth, zenith = load_camera_images(cam_dir)
        cam = Camera(
            name=cam_name,
            longitude=cam_pos["longitude"],
            latitude=cam_pos["latitude"],
            altitude=cam_pos["altitude"],
            image=image,
            azimuth=azimuth,
            zenith=zenith,
        )
        cameras.append(cam)
    return cameras

def load_camera_positions(set_path: Path) -> dict[str, dict]:
    with open(set_path, "r") as f:
        data = f.read()
    # Split the data into sections for each camera
    sections = data.strip().split('----------------------------------')
    # Remove empty sections
    sections = [s.strip() for s in sections if s.strip()]
    positions = {}
    for section in sections:
        # Split section into lines and remove empty lines
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if len(lines) < 4:  # Skip sections that don't have enough data
            continue
        # Extract camera name
        name_line = lines[1].strip()
        # Extract position data
        coords_line = lines[3].strip()
        try:
            # Parse coordinates
            lon, lat, alt = map(float, coords_line.split())
            # Create camera entry
            position = {
                'longitude': lon,
                'latitude': lat,
                'altitude': alt
            }
            positions[name_line] = position
        except (ValueError, IndexError) as e:
            print(f"Error parsing camera data: {e}")
            continue
    return positions

def load_radar_point_cloud(dat_path: Path | str):
    data = np.loadtxt(dat_path, dtype=np.float32)
    data = torch.from_numpy(data)
    alts, lats, lons, dens = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return alts, lats, lons, dens

def load_matrix_data(dat_path: Path | str):
    mat = []
    with open(dat_path, "r") as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            mat.append(row)
    mat = torch.tensor(mat, dtype=torch.float32)
    return mat

def save_matrix_data(tensor: torch.Tensor, dat_path: Path | str):
    with open(dat_path, "w") as f:
        for row in tensor:
            line = " ".join(f"{val:.6f}" for val in row.tolist())
            f.write(line + "\n")

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

def save_3d_grid_data(array: torch.Tensor, file_path: Path):
    array = array.numpy(force=True)
    ni, nj, nk = array.shape
    indices = np.indices((ni, nj, nk)).reshape(3, -1).T  # Generate i, j, k indices efficiently
    values = array.ravel().reshape(-1, 1)  # Flatten array values
    data = np.hstack((indices, values))  # Combine indices with values
    np.savetxt(file_path, data, fmt="%d %d %d %.6f")  # Save to file with formatting

def load_2d_grid_data(dat_path: Path | str):
    data = np.loadtxt(dat_path)
    indices = data[:, :2].astype(int)
    values = data[:, 2]
    ni, nj = indices.max(axis=0) + 1
    array = np.zeros((ni, nj), dtype=values.dtype)
    array[indices[:, 0], indices[:, 1]] = values
    return torch.from_numpy(array).to(torch.float32)

def save_2d_grid_data(array: torch.Tensor, file_path: Path):
    array = array.numpy(force=True)
    ni, nj = array.shape
    indices = np.indices((ni, nj)).reshape(2, -1).T  # Generate i, j indices
    values = array.ravel().reshape(-1, 1)
    data = np.hstack((indices, values))
    np.savetxt(file_path, data, fmt="%d %d %.6f")
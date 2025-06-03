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
from aurora.physics import PhysicalModel
from aurora.frame import ReferenceFrame
from aurora.models import ReferenceFlux

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)


_minmax_tuple = And(Use(lambda lst: (float(lst[0]), float(lst[1])), lambda p: p[0] < p[1]))
_float = Use(float)

def resolve_relative_to_base(target_path: Path, base_path: Path) -> Path:
    base_path = Path(base_path)
    target_path = Path(target_path)
    if target_path.is_absolute():
        return target_path.resolve()
    abs_base = base_path.resolve()
    return (abs_base / target_path).resolve()

def path_validator(base_path: Path):
    return And(
        Use(Path),
        Use(lambda p: resolve_relative_to_base(p, base_path)),
        lambda p: p.exists(),
        error="Must be a valid path"
    )

def parse_physical_model(desc: dict, device: torch.device):
    altitude_bins = load_matrix_data(desc["altitude_bins"]).flatten()
    energy_bins = load_matrix_data(desc["energy_bins"]).flatten()
    emission_matrix = load_matrix_data(desc["emission_matrix"]).T
    density_matrix = load_matrix_data(desc["density_matrix"]).T
    min_altitude = altitude_bins[0].item()
    max_altitude = altitude_bins[-1].item()
    frame_desc = desc["reference_frame"]
    vol_desc = desc["reconstruction_volume"]
    physical_model = PhysicalModel(
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        emission_matrix=emission_matrix,
        density_matrix=density_matrix,
        frame=ReferenceFrame(
            origin_latitude=frame_desc["origin_latitude"],
            origin_longitude=frame_desc["origin_longitude"],
            origin_altitude=min_altitude,
            field_inclination=frame_desc["field_inclination"],
            field_declination=frame_desc["field_declination"],
            range_south=vol_desc["range_south"],
            range_east=vol_desc["range_east"],
            height=max_altitude - min_altitude,
            device=device
        ),
        device=device
    )
    return physical_model

def load_physical_model(yaml_path: Path | str, device: torch.device):
    yaml_path = Path(yaml_path)
    relative_path = path_validator(yaml_path.parent)
    model_desc_schema = Schema({
        Optional("name"): str,
        "altitude_bins": relative_path,
        "energy_bins": relative_path,
        "emission_matrix": relative_path,
        "density_matrix": relative_path,
        "reference_frame": {
            "origin_latitude": _float,
            "origin_longitude": _float,
            "field_inclination": _float,
            "field_declination": _float
        },
        "reconstruction_volume": {
            "range_south": _minmax_tuple,
            "range_east": _minmax_tuple,
        }
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = model_desc_schema.validate(data)
    return parse_physical_model(desc, device)

def parse_reference_flux(desc: dict, device: torch.device):
    return ReferenceFlux(
        image=load_3d_grid_data(desc["flux"]),
        range_south=desc["range_south"],   
        range_east=desc["range_east"],
        device=device
    )

def load_reference_flux(yaml_path: Path | str, device: torch.device):
    yaml_path = Path(yaml_path)
    relative_path = path_validator(yaml_path.parent)
    ref_flux_desc_schema = Schema({
        Optional("name"): str,
        "flux": relative_path,
        "range_south": _minmax_tuple,
        "range_east": _minmax_tuple
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = ref_flux_desc_schema.validate(data)
    return parse_reference_flux(desc, device)

def load_camera_images(cam_dir: Path | str):
    cam_dir = Path(cam_dir)
    image = load_matrix_data(cam_dir / "image.dat")
    azimuth = load_matrix_data(cam_dir / "az_cam.dat")
    zenith = load_matrix_data(cam_dir / "ze_cam.dat")
    return image, azimuth, zenith

def load_radar_point_cloud(dat_path: Path | str):
    data = np.loadtxt(dat_path, dtype=np.float32)
    data = torch.from_numpy(data)
    alts, lats, lons, vals = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return alts, lats, lons, vals

def load_dataset(yaml_path: Path | str) -> list[Camera]:
    yaml_path = Path(yaml_path)
    relative_path = path_validator(yaml_path.parent)
    dataset_desc_schema = Schema({
        Optional("name"): str,
        Optional("cameras"): {
            "positions": relative_path,
            "images": relative_path
        },
        Optional("radar"): relative_path
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = dataset_desc_schema.validate(data)
    cameras, radar_data = None, None
    if "cameras" in desc:
        positions = load_camera_positions(desc["cameras"]["positions"])
        images_dir = desc["cameras"]["images"]
        cameras = []
        for cam_name, cam_pos in positions.items():
            cam_dir = images_dir / cam_name
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
    if "radar" in desc:
        radar_data = load_radar_point_cloud(desc["radar"])
    return cameras, radar_data

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
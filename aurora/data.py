from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, List, Dict, Tuple

from schema import Schema, Optional as Option, And, Or, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper
import torch
import numpy as np

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox


PathLike = Union[Path, str]


def load_yaml(file_path: PathLike, schema: Optional[Schema] = None) -> Dict:
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    if schema is not None:
        return schema.validate(data)
    else:
        return data

def save_yaml(file_path: PathLike, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)

def resolve_relative_to_base(target_path: PathLike, base_path: PathLike) -> Path:
    base_path = Path(base_path)
    target_path = Path(target_path)
    if target_path.is_absolute():
        return target_path.resolve()
    abs_base = base_path.resolve()
    return (abs_base / target_path).resolve()

def path_validator(base_path: Optional[PathLike] = None):
    ops = [Use(Path)]
    if base_path is not None:
        ops.append(Use(lambda p: resolve_relative_to_base(p, base_path)))
    ops.append(lambda p: p.exists())
    return And(*ops, error="Must be a valid path")

@dataclass
class PhysicalModel:
    emis_mat: torch.Tensor
    dens_mat: torch.Tensor
    altitude_bins: torch.Tensor
    energy_bins: torch.Tensor

def physical_model_schema(base_path=None):
    valid_path = path_validator(base_path)
    return Schema({
        "emis_mat": valid_path,
        "dens_mat": valid_path,
        "altitude_bins": valid_path,
        "energy_bins": valid_path,
    })

def load_physical_model(yaml_path: PathLike, device=torch.device("cpu")):
    yaml_path = Path(yaml_path)
    schema = physical_model_schema(yaml_path.parent)
    data = load_yaml(yaml_path, schema)
    return physical_model_from_dict(data, device)

def physical_model_from_dict(data: Dict, device=torch.device("cpu")):
    return PhysicalModel(
        emis_mat=load_emission_matrix(data["emis_mat"]).to(device),
        dens_mat=load_density_matrix(data["dens_mat"]).to(device),
        altitude_bins=load_altitude_bins(data["altitude_bins"]).to(device),
        energy_bins=load_energy_bins(data["energy_bins"]).to(device),
    )

def frame_schema(base_path=None):
    return Schema({
        "origin_lat": Use(float),
        "origin_lon": Use(float),
        "origin_alt": Use(float),
        "field_inc": Use(float),
        "field_dec": Use(float),
    })

def load_frame(yaml_path: PathLike, device=torch.device("cpu")):
    schema = frame_schema(yaml_path.parent)
    data = load_yaml(yaml_path, schema)
    return frame_from_dict(data, device)

def frame_from_dict(data: Dict, device=torch.device("cpu")):
    return Frame(
        origin_latitude=data["origin_lat"],
        origin_longitude=data["origin_lon"],
        origin_altitude=data["origin_alt"],
        field_inclination=data["field_inc"],
        field_declination=data["field_dec"],
    ).to(device)

def frame_to_dict(frame: Frame):
    return {
        "origin_lat": frame.origin_latitude,
        "origin_lon": frame.origin_latitude,
        "origin_alt": frame.origin_latitude,
        "field_inc": frame.field_inclination,
        "field_dec": frame.field_declination
    }

def bbox_schema(base_path=None):
    valid_path = path_validator(base_path)
    return Schema({
        Option("frame"): Or(frame_schema(base_path), valid_path),
        "east_min": Use(float),
        "east_max": Use(float),
        "south_min": Use(float),
        "south_max": Use(float),
        "alt_min": Use(float),
        "alt_max": Use(float),
    })

def load_bbox(yaml_path: PathLike, frame: Optional[Frame] = None):
    schema = bbox_schema(yaml_path.parent)
    data = load_yaml(yaml_path, schema)
    return bbox_from_dict(data, frame)

def bbox_from_dict(data: Dict, frame: Optional[Frame] = None):
    if frame is None and "frame" in data:
        if isinstance(data["frame"], dict):
            frame = frame_from_dict(data["frame"])
        else:
            frame = load_frame(data["frame"])
    if frame is None:
        raise ValueError("Missing frame description")
    return BBox(
        frame=frame,
        south_range=(data["south_min"], data["south_max"]),
        east_range=(data["east_min"], data["east_max"]),
        altitude_range=(data["alt_min"], data["alt_max"])
    )

def bbox_to_dict(bbox: BBox, frame: Optional[Union[Frame, PathLike]] = None):
    bbox_data = {
        "east_min": bbox.east_range[0],
        "east_max": bbox.east_range[1],
        "south_min": bbox.south_range[0],
        "south_max": bbox.south_range[1],
        "alt_min": bbox.altitude_range[0],
        "alt_max": bbox.altitude_range[1]
    }
    if frame is None:
        return bbox_data
    if isinstance(frame, Frame):
        bbox_data["frame"] = frame_to_dict(frame)
    else:
        bbox_data["frame"] = str(Path(frame))
    return bbox_data

@dataclass
class Config:
    frame: Frame
    bbox: BBox
    physics: PhysicalModel

def config_schema(base_path=None):
    return Schema({
        "frame": frame_schema(base_path),
        "bbox": bbox_schema(base_path),
        "physics": physical_model_schema(base_path)
    }, ignore_extra_keys=True)

def load_config(yaml_path: PathLike, device=torch.device("cpu")):
    schema = config_schema(yaml_path.parent)
    data = load_yaml(yaml_path, schema)
    return config_from_dict(data, device)

def config_from_dict(data: dict, device=torch.device("cpu")):
    frame = frame_from_dict(data["frame"], device)
    bbox = bbox_from_dict(data["bbox"], frame)
    physics = physical_model_from_dict(data["physics"], device)
    return Config(frame, bbox, physics)

def load_camera_images(cam_dir: PathLike, device=torch.device("cpu")):
    cam_dir = Path(cam_dir)
    image = load_matrix_data(cam_dir / "image.dat").to(device)
    azimuth = load_matrix_data(cam_dir / "az_cam.dat").to(device)
    zenith = load_matrix_data(cam_dir / "ze_cam.dat").to(device)
    return image, azimuth, zenith

def load_cameras(cam_pos_set: PathLike, cam_images_dir: PathLike, device=torch.device("cpu")) -> List[Camera]:
    cam_pos_set = Path(cam_pos_set)
    cam_images_dir = Path(cam_images_dir)
    positions = load_camera_positions(cam_pos_set)
    cameras = []
    for cam_name, cam_pos in positions.items():
        cam_dir = cam_images_dir / cam_name
        image, azimuth, zenith = load_camera_images(cam_dir, device)
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

def load_camera_positions(set_path: Path) -> Dict[str, dict]:
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

@dataclass
class RadarData:
    latitudes: torch.Tensor
    longitudes: torch.Tensor
    altitudes: torch.Tensor
    densities: torch.Tensor

def load_radar_point_cloud(dat_path: PathLike):
    data = np.loadtxt(dat_path, dtype=np.float32)
    data = torch.from_numpy(data)
    alts, lats, lons, dens = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    return RadarData(lats, lons, alts, dens)

def load_emission_matrix(dat_path: PathLike):
    return load_matrix_data(dat_path).T

def load_density_matrix(dat_path: PathLike):
    return load_matrix_data(dat_path).T.square()

def load_altitude_bins(dat_path: PathLike):
    return load_matrix_data(dat_path).flatten()

def load_energy_bins(dat_path: PathLike):
    return load_matrix_data(dat_path).flatten()

def load_matrix_data(dat_path: PathLike):
    mat = []
    with open(dat_path, "r") as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            mat.append(row)
    mat = torch.tensor(mat, dtype=torch.float32)
    return mat

def save_matrix_data(tensor: torch.Tensor, dat_path: PathLike):
    with open(dat_path, "w") as f:
        for row in tensor:
            line = " ".join(f"{val:.6f}" for val in row.tolist())
            f.write(line + "\n")

def load_3d_grid_data(dat_path: PathLike):
    data = np.loadtxt(dat_path)
    indices = data[:, :3].astype(int)
    values = data[:, 3]
    # Determine array shape from max index values
    ni, nj, nk = indices.max(axis=0) + 1
    array = np.zeros((ni, nj, nk), dtype=values.dtype)
    # Assign values
    array[indices[:, 0], indices[:, 1], indices[:, 2]] = values
    return torch.from_numpy(array).to(torch.float32)

def save_3d_grid_data(array: torch.Tensor, file_path: PathLike):
    array = array.numpy(force=True)
    ni, nj, nk = array.shape
    indices = np.indices((ni, nj, nk)).reshape(3, -1).T  # Generate i, j, k indices efficiently
    values = array.ravel().reshape(-1, 1)  # Flatten array values
    data = np.hstack((indices, values))  # Combine indices with values
    np.savetxt(file_path, data, fmt="%d %d %d %.6f")  # Save to file with formatting

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
    indices = np.indices((ni, nj)).reshape(2, -1).T  # Generate i, j indices
    values = array.ravel().reshape(-1, 1)
    data = np.hstack((indices, values))
    np.savetxt(file_path, data, fmt="%d %d %.6f")

from pathlib import Path
from dataclasses import dataclass
from typing import Union, Optional, List, Dict

from schema import Schema, Optional as And, Or, Use
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
from aurora.models import ModelConfig


PathLike = Union[Path, str]


def load_yaml(file_path: PathLike) -> Dict:
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=Loader)
    return data

def save_yaml(file_path: PathLike, data):
    with open(file_path, "w") as f:
        yaml.dump(data, f, Dumper=Dumper)

def resolve_relative_to_base(target_path: PathLike, base_path: PathLike) -> Path:
    """
    Return an absolute, resolved Path for `target_path`; if it is relative, interpret
    it relative to `base_path`.
    """
    target_path = Path(target_path)
    if target_path.is_absolute():
        return target_path.resolve()
    base_path = Path(base_path)
    abs_base = base_path.resolve()
    abs_target = abs_base / target_path
    return abs_target.resolve()

def path_validator(base_path: Optional[PathLike] = None):
    """
    Creates schema validator that ensures a given path is a valid sub-path of the
    provided base_path, if not absolute.
    """
    validators = [Use(Path)]
    if base_path is not None:
        validators.append(Use(lambda p: resolve_relative_to_base(p, base_path)))
    validators.append(lambda p: p.exists())
    return And(*validators)


def config_schema(base_path=None):
    valid_path = path_validator(base_path)
    return Schema({
        "frame": {
            "origin_lat": Use(float),
            "origin_lon": Use(float),
            "origin_alt": Use(float),
            "field_inc": Use(float),
            "field_dec": Use(float),
        },
        "bbox": {
            "east_min": Use(float),
            "east_max": Use(float),
            "south_min": Use(float),
            "south_max": Use(float),
            "alt_min": Use(float),
            "alt_max": Use(float),
        },
        "physics": {
            "emis_mat": Or(valid_path, {str: valid_path}, only_one=True),
            "dens_mat": valid_path,
            "altitude_bins": valid_path,
            "energy_bins": valid_path,
        }
    }, ignore_extra_keys=True)

def load_config(yaml_path: PathLike):
    yaml_path = Path(yaml_path)
    schema = config_schema(yaml_path.parent)
    data = load_yaml(yaml_path)
    data = schema.validate(data)
    return config_from_dict(data)

def config_from_dict(data: dict):
    frame_data = data["frame"]
    frame = Frame(
        origin_latitude=frame_data["origin_lat"],
        origin_longitude=frame_data["origin_lon"],
        origin_altitude=frame_data["origin_alt"],
        field_inclination=frame_data["field_inc"],
        field_declination=frame_data["field_dec"]
    )
    
    bbox_data = data["bbox"]
    bbox = BBox(
        south_range=(bbox_data["south_min"], bbox_data["south_max"]),
        east_range=(bbox_data["east_min"], bbox_data["east_max"]),
        altitude_range=(bbox_data["alt_min"], bbox_data["alt_max"]),
        frame=frame
    )
    
    phys_data = data["physics"]
    altitude_bins = load_altitude_bins(phys_data["altitude_bins"])
    energy_bins = load_energy_bins(phys_data["energy_bins"])

    emis_mats = None
    if "emis_mat" in phys_data:
        if isinstance(phys_data["emis_mat"],  Path):
            emis_mats = load_emission_matrix(phys_data["emis_mat"])
        else:
            emis_mats = {}
            for wl, emis_dat_path in phys_data["emis_mat"].items():
                emis_mats[wl] = load_emission_matrix(emis_dat_path)
    
    dens_mat = None
    if "dens_mat" in phys_data:
        dens_mat = load_density_matrix(phys_data["dens_mat"])
    
    return ModelConfig(
        bbox=bbox,
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        emis_mats=emis_mats,
        dens_mat=dens_mat
    )


@dataclass
class CameraInfo:
    camera_id: int
    name: str
    longitude: float
    latitude: float
    altitude: float
    location_name: str
    wavelength: Optional[str] = None

def load_cameras(set_path: PathLike, imgs_dir: PathLike, wl_filter: Optional[List[str]] = None) -> List[Camera]:
    """
    Loads cameras from a position description file and image data directory.
    
    Args:
        set_path: Path to camera position .set file.
        imgs_dir: Path to the directory containing camera image data. The directory
            structure is assumed to be either imgs_dir/{camera_location}/*.dat for
            single wavelength or imgs_dir/{camera_location}/{wavelength}/*.dat for
            multiple wavelengths
        wl_filter (optional): List restricting the camera wavelengths to be loaded.
            If None, all available camera data is loaded.
    
    Returns:
        List of cameras.
    
    Raises:
        FileNotFoundError: If the image directory or .set file path doesn't exist.
    """
    imgs_dir = Path(imgs_dir)
    if not imgs_dir.exists():
        raise FileNotFoundError(f"Camera image directory {imgs_dir} not found.")
    
    infos = load_camera_positions(set_path)
    if wl_filter is not None:
        infos = filter(lambda c: c.wavelength in wl_filter, infos)
    
    cameras = []
    for info in infos:
        cam_dir = imgs_dir / info.location_name
        if info.wavelength is not None:
            cam_dir = cam_dir / info.wavelength
        if not cam_dir.exists():
            print(f"Warning: couldn't find images for camera {info.camera_id}.")
            continue
        cam = load_camera(info, cam_dir)
        cameras.append(cam)
    return cameras

def load_camera(cam_info: CameraInfo, cam_dir: PathLike):
    """
    Loads a single camera using the provided information in `cam_info` and loading
    images `image.dat`, `az_cam.dat` and `ze_cam.dat` from `cam_dir`.
    """
    cam_dir = Path(cam_dir)
    image = load_matrix_data(cam_dir / "image.dat")
    azimuth = load_matrix_data(cam_dir / "az_cam.dat")
    zenith = load_matrix_data(cam_dir / "ze_cam.dat")
    
    return Camera(
        camera_id=cam_info.camera_id,
        name=cam_info.name,
        longitude=cam_info.longitude,
        latitude=cam_info.latitude,
        altitude=cam_info.altitude,
        location_name=cam_info.location_name,
        image=image,
        azimuth=azimuth,
        zenith=zenith,
        wavelength=cam_info.wavelength
    )

def load_camera_positions(set_path: PathLike) -> List[CameraInfo]:
    """
    Load camera positions and information from a .set file.
    
    Args:
        set_path (PathLike): Path to the .set file
        
    Returns:
        List of CameraInfo objects parsed from the .set file.
        
    Raises:
        ValueError: If a camera section is missing required data.
    """
    with open(set_path, "r") as f:
        data = f.read()
    
    # Split the data into sections for each camera
    sections = data.strip().split('----------------------------------')
    sections = [s.strip() for s in sections if s.strip()]
    
    cameras = []
    
    for idx, section in enumerate(sections, start=1):
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        
        if len(lines) < 4:
            continue
        
        # Parse the section
        location_name = None
        wavelength = None
        coords = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            # Look for position name
            if 'position name' in line.lower():
                if i + 1 < len(lines):
                    location_name = lines[i + 1].strip()
                    i += 2
                    continue
            # Look for coordinates
            if 'longitude' in line.lower() and 'latitude' in line.lower():
                if i + 1 < len(lines):
                    coords_line = lines[i + 1].strip()
                    try:
                        lon, lat, alt = map(float, coords_line.split())
                        coords = (lon, lat, alt)
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Error parsing coordinates in section {idx}: {e}")
                    i += 2
                    continue
            # Look for wavelength
            if 'wavelength' in line.lower():
                if i + 1 < len(lines):
                    try:
                        wavelength = lines[i + 1].strip()
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Error parsing wavelength in section {idx}: {e}")
                    i += 2
                    continue
            i += 1
        
        # Validate required fields
        if coords is None:
            raise ValueError(f"Section {idx} missing coordinates")
        if location_name is None:
            raise ValueError(f"Section {idx} missing location name")
        
        # Choose a unique name
        cam_name = location_name
        if wavelength is not None:
            cam_name = f"{location_name}-{wavelength}"
        
        # Create CameraData object
        camera = CameraInfo(
            camera_id=idx,
            name=cam_name,
            longitude=coords[0],
            latitude=coords[1],
            altitude=coords[2],
            location_name=location_name,
            wavelength=wavelength
        )
        cameras.append(camera)
    return cameras


@dataclass
class RadarPointCloud:
    latitudes: torch.Tensor
    longitudes: torch.Tensor
    altitudes: torch.Tensor
    densities: torch.Tensor

def load_radar_point_cloud(dat_path: PathLike):
    data = np.loadtxt(dat_path, dtype=np.float32)
    data = torch.from_numpy(data) # (n, 4)
    h, lat, lon, d = data.T # (4, n)
    return RadarPointCloud(lat, lon, h, d)


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

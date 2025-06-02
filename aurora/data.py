from pathlib import Path

from schema import Schema, Optional, And, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import torch

from aurora.camera import Camera
from aurora.physics import PhysicalModel
from aurora.frame import ReferenceFrame
from aurora.models import ReferenceFlux
from aurora.utils import load_matrix_data, load_3d_grid_data

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)


_minmax = And(Use(lambda lst: (float(lst[0]), float(lst[1])), lambda p: p[0] < p[1]))
_float = Use(float)
_path = And(Use(Path), lambda p: p.exists(), error="Must be a valid path")

_model_desc_schema = Schema({
    Optional("name"): str,
    "altitude_bins": _path,
    "energy_bins": _path,
    "emission_matrix": _path,
    "reference_frame": {
        "origin_latitude": _float,
        "origin_longitude": _float,
        "field_inclination": _float,
        "field_declination": _float
    },
    "reconstruction_volume": {
        "range_south": _minmax,
        "range_east": _minmax,
    }
})

_dataset_desc_schema = Schema({
    Optional("name"): str,
    "positions": _path,
    "images": _path
})

_ref_flux_desc_schema = Schema({
    Optional("name"): str,
    "flux": _path,
    "range_south": _minmax,
    "range_east": _minmax
})

def parse_physical_model(desc: dict, device: torch.device):
    altitude_bins=load_matrix_data(desc["altitude_bins"]).flatten()
    energy_bins=load_matrix_data(desc["energy_bins"]).flatten()
    emission_matrix=load_matrix_data(desc["emission_matrix"]).T
    frame_desc = desc["reference_frame"]
    vol_desc = desc["reconstruction_volume"]
    frame = ReferenceFrame(
        origin_latitude=frame_desc["origin_latitude"],
        origin_longitude=frame_desc["origin_longitude"],
        origin_altitude=altitude_bins[0].item(),
        field_inclination=frame_desc["field_inclination"],
        field_declination=frame_desc["field_declination"],
        range_south=vol_desc["range_south"],
        range_east=vol_desc["range_east"],
        height=(altitude_bins[-1] - altitude_bins[0]).item(),
        device=device
    )
    pm = PhysicalModel(
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        emission_matrix=emission_matrix,
        frame=frame,
        device=device
    )
    return pm

def load_physical_model(yaml_path: Path | str, device: torch.device):
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = _model_desc_schema.validate(data)
    return parse_physical_model(desc, device)

def parse_reference_flux(desc: dict, device: torch.device):
    return ReferenceFlux(
        image=load_3d_grid_data(desc["flux"]),
        range_south=desc["range_south"],   
        range_east=desc["range_east"],
        device=device
    )

def load_reference_flux(yaml_path: Path | str, device: torch.device):
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = _ref_flux_desc_schema.validate(data)
    return parse_reference_flux(desc, device)

def load_cameras(yaml_path: Path | str) -> list[Camera]:
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = _dataset_desc_schema.validate(data)
    positions = load_camera_positions(desc["positions"])
    images_dir = desc["images"]
    cameras = []
    for cam_name, cam_pos in positions.items():
        cam_dir = images_dir / cam_name
        cam = Camera(
            name=cam_name,
            longitude=cam_pos["longitude"],
            latitude=cam_pos["latitude"],
            altitude=cam_pos["altitude"],
            image=load_matrix_data(cam_dir / "image.dat"),
            azimuth=load_matrix_data(cam_dir / "az_cam.dat"),
            zenith=load_matrix_data(cam_dir / "ze_cam.dat"),
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

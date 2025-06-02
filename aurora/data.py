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
    yaml_path = Path(yaml_path)
    relative_path = path_validator(yaml_path.parent)
    model_desc_schema = Schema({
        Optional("name"): str,
        "altitude_bins": relative_path,
        "energy_bins": relative_path,
        "emission_matrix": relative_path,
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
        "range_south": _minmax,
        "range_east": _minmax
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = ref_flux_desc_schema.validate(data)
    return parse_reference_flux(desc, device)

def parse_camera(cam_pos: dict, cam_name: str, cam_dir: Path | str):
    if not isinstance(cam_dir, Path):
        cam_dir = Path(cam_dir)
    return Camera(
        name=cam_name,
        longitude=cam_pos["longitude"],
        latitude=cam_pos["latitude"],
        altitude=cam_pos["altitude"],
        image=load_matrix_data(cam_dir / "image.dat"),
        azimuth=load_matrix_data(cam_dir / "az_cam.dat"),
        zenith=load_matrix_data(cam_dir / "ze_cam.dat"),
    )

def load_cameras(yaml_path: Path | str) -> list[Camera]:
    yaml_path = Path(yaml_path)
    relative_path = path_validator(yaml_path.parent)
    dataset_desc_schema = Schema({
        Optional("name"): str,
        "positions": relative_path,
        "images": relative_path
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = dataset_desc_schema.validate(data)
    positions = load_camera_positions(desc["positions"])
    images_dir = desc["images"]
    cameras = []
    for cam_name, cam_pos in positions.items():
        cam_dir = images_dir / cam_name
        cam = parse_camera(cam_pos, cam_name, cam_dir)
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

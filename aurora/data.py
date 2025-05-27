from pathlib import Path
from schema import Schema, Optional, And, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import torch

from aurora.camera import Camera
from aurora.physics import PhysicalModel, ReferenceFrame, ReferenceFlux
from aurora.utils import load_matrix_data, load_3d_grid_data, to_minmax

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)


def load_physical_model(yaml_path: Path | str, device: torch.device):
    minmax = Use(to_minmax)
    as_float = Use(float)
    path = And(Use(Path), lambda p: p.exists(), error="Must be a valid path")
    schema = Schema({
        Optional("name"): str,
        "reference_frame": {
            "origin_latitude": as_float,
            "origin_longitude": as_float,
            "origin_altitude": as_float,
            "field_inclination": as_float,
            "field_declination": as_float
        },
        "altitude_bins": path,
        "energy_bins": path,
        "emission_matrix": path,
        "reconstruction_volume": {
            "oblique_range_x": minmax,
            "oblique_range_y": minmax,
            "oblique_height": as_float
        },
        Optional("reference_flux"): {
            "flux": path,
            "oblique_range_x": minmax,
            "oblique_range_y": minmax
        }
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = schema.validate(data)
    frame_desc = desc["reference_frame"]
    vol_desc = desc["reconstruction_volume"]
    frame = ReferenceFrame(
        origin_latitude=frame_desc["origin_latitude"],
        origin_longitude=frame_desc["origin_longitude"],
        origin_altitude=frame_desc["origin_altitude"],
        field_inclination=frame_desc["field_inclination"],
        field_declination=frame_desc["field_declination"],
        oblique_range_x=vol_desc["oblique_range_x"],
        oblique_range_y=vol_desc["oblique_range_y"],
        oblique_height=vol_desc["oblique_height"],
        device=device
    )
    pm = PhysicalModel(
        altitude_bins=load_matrix_data(desc["altitude_bins"]).flatten(),
        energy_bins=load_matrix_data(desc["energy_bins"]).flatten(),
        emission_matrix=load_matrix_data(desc["emission_matrix"]).T,
        frame=frame,
        device=device
    )
    ref_flux = None
    if "reference_flux" in desc:
        flux_desc = desc["reference_flux"]
        ref_flux = ReferenceFlux(
            flux=load_3d_grid_data(flux_desc["flux"]),
            oblique_range_x=flux_desc["oblique_range_x"],   
            oblique_range_y=flux_desc["oblique_range_y"],
            device=device
        )
    return pm, ref_flux

def load_cameras(yaml_path: Path | str) -> list[Camera]:
    path = And(Use(Path), lambda p: p.exists(), error="Must be a valid path")
    schema = Schema({
        Optional("name"): str,
        "positions": path,
        "images": path
    })
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = schema.validate(data)
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

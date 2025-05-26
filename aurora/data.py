from pathlib import Path
import numpy as np
from schema import Schema, Optional, And, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
from dataclasses import dataclass
from typing import NamedTuple

from aurora.camera import Camera

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)



class MinMax(NamedTuple):
    min: float
    max: float

def to_minmax(lst):
    if not isinstance(lst, (list, tuple)):
        raise TypeError("Value must be a list or tuple")
    if len(lst) != 2:
        raise ValueError("List must have exactly two elements")
    return MinMax(float(lst[0]), float(lst[1]))

@dataclass
class ReferenceFrameDescription:
    origin_latitude: float
    origin_longitude: float
    origin_altitude: float
    field_inclination: float
    field_declination: float

@dataclass
class VolumeDescription:
    oblique_range_x: MinMax
    oblique_range_y: MinMax
    oblique_height: float

@dataclass
class PhysicalModelDescription:
    altitude_bins: np.ndarray
    energy_bins: np.ndarray
    emission_matrix: np.ndarray
    reference_frame: ReferenceFrameDescription
    reconstruction_volume: VolumeDescription

@dataclass
class ReferenceFlux:
    image: np.ndarray
    oblique_range_x: MinMax
    oblique_range_y: MinMax


def load_dataset_description(yaml_path: Path) -> tuple[list[Camera], PhysicalModelDescription]:
    minmax = Use(to_minmax)
    as_float = Use(float)
    path = And(Use(Path), lambda p: p.exists(), error="Must be a valid path")
    schema = Schema({
        Optional("name"): str,
        "cameras": {
            "positions": path,
            "images": path
        },
        "model": {
            "altitude_bins": path,
            "energy_bins": path,
            "emission_matrix": path,
            "reference_frame": {
                "origin_latitude": as_float,
                "origin_longitude": as_float,
                "origin_altitude": as_float,
                "field_inclination": as_float,
                "field_declination": as_float
            },
            "reconstruction_volume": {
                "oblique_range_x": minmax,
                "oblique_range_y": minmax,
                "oblique_height": as_float
            },
        },
        Optional("reference_flux"): {
            "image": path,
            "oblique_range_x": minmax,
            "oblique_range_y": minmax
        }
    })
    
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = schema.validate(data)
    cameras = parse_dataset_cameras(desc["cameras"])
    model = parse_physical_model_description(desc["model"])
    return cameras, model

def parse_dataset_cameras(desc: dict) -> list[Camera]:
    camera_positions = parse_camera_positions(desc["positions"])
    images_dir = desc["images"]
    cameras = []
    for cam_name, cam_pos in camera_positions.items():
        camera = Camera(
            name=cam_name,
            longitude=cam_pos["longitude"],
            latitude=cam_pos["latitude"],
            altitude=cam_pos["altitude"],
            image=parse_matrix_data(images_dir / cam_name / "image.dat"),
            azimuth=parse_matrix_data(images_dir / cam_name / "az_cam.dat"),
            zenith=parse_matrix_data(images_dir / cam_name / "ze_cam.dat"),
        )
        cameras.append(camera)
    return cameras

def parse_camera_positions(set_path: Path) -> dict[str, dict]:
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


def parse_matrix_data(dat_path: Path):
    mat = []
    with open(dat_path, "r") as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            mat.append(row)
    mat = np.array(mat, dtype=np.float32)
    return mat


def parse_physical_model_description(desc: dict):
    frame = desc["reference_frame"]
    volume = desc["reconstruction_volume"]
    return PhysicalModelDescription(
        altitude_bins=parse_matrix_data(desc["altitude_bins"]).flatten(),
        energy_bins=parse_matrix_data(desc["energy_bins"]).flatten(),
        emission_matrix=parse_matrix_data(desc["emission_matrix"]).T,
        reference_frame=ReferenceFrameDescription(
            origin_latitude=frame["origin_latitude"],
            origin_longitude=frame["origin_longitude"],
            origin_altitude=frame["origin_altitude"],
            field_inclination=frame["field_inclination"],
            field_declination=frame["field_declination"]
        ),
        reconstruction_volume=VolumeDescription(
            oblique_range_x=volume["oblique_range_x"],
            oblique_range_y=volume["oblique_range_y"],
            oblique_height=volume["oblique_height"]
        )
    )

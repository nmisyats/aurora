from pathlib import Path
import os
import numpy as np
from dataclasses import dataclass
from schema import Schema, Optional, And, Use
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

def load_yaml(stream):
    return yaml.load(stream, Loader=Loader)


@dataclass
class Camera:
    name: str
    latitude: float
    longitude: float
    altitude: float
    image: np.ndarray
    azimuth: np.ndarray
    zenith: np.ndarray

    def __repr__(self):
        return ", ".join(["Camera=(",
            f"name={self.name}",
            f"latitude={self.latitude}",
            f"longitude={self.longitude}",
            f"altitude={self.altitude}",
            f"image={type(self.image)}",
            f"azimuth={type(self.azimuth)}",
            f"zenith={type(self.zenith)}",
        ")"])
    
    def __str__(self):
        return self.__repr__(self)

@dataclass
class Direction:
    inc: float
    dec: float

@dataclass
class XY:
    x: float
    y: float

@dataclass
class XYZ:
    x: float
    y: float
    z: float

@dataclass
class Volume:
    lat: float
    lon: float
    alt: float
    min: XYZ
    max: XYZ

@dataclass
class Q0:
    image: np.ndarray
    min: XY
    max: XY

@dataclass
class Model:
    altitude_bins: np.ndarray
    energy_bins: np.ndarray
    emission_matrix: np.ndarray
    field: Direction
    volume: Volume
    q0: Q0 | None = None


def to_float_tuple(n):
    def convert(t):
        if not isinstance(t, (tuple, list)):
            raise TypeError("Value must be a tuple or list")
        if len(t) != n:
            raise ValueError(f"Tuple must have exactly {n} elements")
        return tuple(map(float, t))
    return convert

def load_dataset_description(yaml_path: Path) -> tuple[list[Camera], Model]:
    minmax = Use(to_float_tuple(2))
    path = And(Use(Path), lambda p: p.exists(), error="Must be a valid and existing path")
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
            "magnetic_field": {
                "inclination": Use(float),
                "declination": Use(float)
            },
            "reconstruction_volume": {
                "latitude": Use(float),
                "longitude": Use(float),
                "altitude": Use(float),
                "range_x": minmax,
                "range_y": minmax,
                "range_z": minmax
            },
            Optional("reference_q0"): {
                "image": path,
                "range_x": minmax,
                "range_y": minmax
            }
        },
    })
    
    with open(yaml_path, "r") as f:
        data = load_yaml(f)
        desc = schema.validate(data)
    cameras = parse_dataset_cameras(desc["cameras"])
    model = parse_physical_model(desc["model"])
    return cameras, model

def parse_dataset_cameras(desc: dict) -> list[Camera]:
    camera_positions = parse_camera_positions(desc["positions"])
    images_dir = desc["images"]
    cameras = []
    for cam_name, cam_pos in camera_positions.items():
        image = parse_matrix_data(images_dir / cam_name / "image.dat")
        azimuth = parse_matrix_data(images_dir / cam_name / "az_cam.dat")
        zenith = parse_matrix_data(images_dir / cam_name / "ze_cam.dat")
        camera = Camera(
            name=cam_name,
            longitude=cam_pos["longitude"],
            latitude=cam_pos["latitude"],
            altitude=cam_pos["altitude"],
            image=image,
            azimuth=azimuth,
            zenith=zenith,
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
    img = []
    with open(dat_path, "r") as f:
        for line in f:
            row = [float(num) for num in line.strip().split()]
            img.append(row)
    img = np.array(img, dtype=np.float32)
    return img


def parse_physical_model(desc: dict):
    field = desc["magnetic_field"]
    vol = desc["reconstruction_volume"]
    x_min, x_max = vol["range_x"]
    y_min, y_max = vol["range_y"]
    z_min, z_max = vol["range_z"]
    pm = Model(
        altitude_bins=parse_matrix_data(desc["altitude_bins"]).flatten(),
        energy_bins=parse_matrix_data(desc["energy_bins"]).flatten(),
        emission_matrix=parse_matrix_data(desc["emission_matrix"]).T,
        field=Direction(
            field["inclination"],
            field["declination"]),
        volume=Volume(
            lat=vol["latitude"],
            lon=vol["longitude"],
            alt=vol["altitude"],
            min=XYZ(x_min, y_min, z_min),
            max=XYZ(x_max, y_max, z_max)
        )
    )
    if "reference_q0" in desc:
        q0 = desc["reference_q0"]
        x_min, x_max = q0["range_x"]
        y_min, y_max = q0["range_y"]
        pm.q0 = Q0(
            image=parse_matrix_data(q0["image"]),
            min=XY(x_min, y_min),
            max=XY(x_max, y_max)
        )
    return pm

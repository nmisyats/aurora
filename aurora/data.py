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
    longitude: float
    latitude: float
    altitude: float
    image: np.ndarray
    azimuth: np.ndarray
    zenith: np.ndarray


@dataclass
class Model:
    altitude_bins: np.ndarray
    energy_bins: np.ndarray
    emission_matrix: np.ndarray
    mag_field_inclination: float
    mag_field_declination: float
    box_origin_lat: float
    box_origin_lon: float
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]
    has_ref_q0: bool = False
    ref_q0_image: np.ndarray | None = None
    ref_q0_range_min: tuple[float, float] | None = None
    ref_q0_range_max: tuple[float, float] | None = None
    ref_q0_range_scale: tuple[float, float] | None = None


def load_dataset_description(yaml_path: Path) -> tuple[list[Camera], Model]:
    schema = Schema({
        Optional("name"): str,
        "cameras": {
            "positions": And(Use(Path), lambda p: p.exists()),
            "images": And(Use(Path), lambda p: p.exists())
        },
        "model": {
            "M_emis": And(Use(Path), lambda p: p.exists()),
            "altitude_bins": And(Use(Path), lambda p: p.exists()),
            "energy_bins": And(Use(Path), lambda p: p.exists()),
            "mag_field": {
                "inclination": float,
                "declination": float
            },
            "volume": {
                "origin": {"lat": float, "lon": float},
                "range_x": {"min": float, "max": float},
                "range_y": {"min": float, "max": float},
                "range_z": {"min": float, "max": float}
            },
            Optional("ref_q0"): {
                "image": Use(Path),
                "range_x": {
                    "min": float,
                    "max": float,
                    Optional("scale"): float
                },
                "range_y": {
                    "min": float,
                    "max": float,
                    Optional("scale"): float
                }
            }
        }
    })
    
    with open(yaml_path, "r") as f:
        desc = schema.validate(load_yaml(f))
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
    altitude_bins = parse_matrix_data(desc["altitude_bins"]).flatten()
    energy_bins = parse_matrix_data(desc["energy_bins"]).flatten()
    m_emis = parse_matrix_data(desc["M_emis"]).T
    vol = desc["volume"]
    x, y, z = vol["range_x"], vol["range_y"], vol["range_z"]
    pm = Model(
        altitude_bins=altitude_bins,
        energy_bins=energy_bins,
        emission_matrix=m_emis,
        mag_field_inclination=desc["mag_field"]["inclination"],
        mag_field_declination=desc["mag_field"]["declination"],
        box_origin_lat=vol["origin"]["lat"],
        box_origin_lon=vol["origin"]["lon"],
        box_min=(x["min"], y["min"], z["min"]),
        box_max=(x["max"], y["max"], z["max"]),
    )
    if "ref_q0" in desc:
        ref = desc["ref_q0"]
        x, y = ref["range_x"], ref["range_y"]
        pm.has_ref_q0 = True
        pm.ref_q0_image = parse_matrix_data(ref["image"])
        pm.ref_q0_range_min = (x["min"], y["min"])
        pm.ref_q0_range_max = (x["max"], y["max"])
        pm.ref_q0_range_scale = (x.get("scale", 1.0), y.get("scale", 1.0))
    return pm

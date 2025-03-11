from pathlib import Path
import os
import numpy as np
from dataclasses import dataclass


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
    altitudes: np.ndarray
    energies: np.ndarray
    matrix: np.ndarray
    field_azimuth: float
    field_elevation: float
    box_min: tuple[float, float, float]
    box_max: tuple[float, float, float]


def parse_dataset_cameras(dataset_path: Path) -> dict[str, Camera]:
    cam_dirs = [entry.name for entry in os.scandir(dataset_path) if entry.is_dir()]
    
    camera_positions = parse_camera_positions(dataset_path / "camera_position.set")
    cameras = {}
    for cam_name in cam_dirs:
        image = parse_matrix_data(dataset_path / cam_name / "image.dat")
        azimuth = parse_matrix_data(dataset_path / cam_name / "az_cam.dat")
        zenith = parse_matrix_data(dataset_path / cam_name / "ze_cam.dat")
        camera = Camera(
            name=cam_name,
            longitude=camera_positions[cam_name]["longitude"],
            latitude=camera_positions[cam_name]["latitude"],
            altitude=camera_positions[cam_name]["altitude"],
            image=image,
            azimuth=azimuth,
            zenith=zenith,
        )
        cameras[camera.name] = camera
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
    img = np.array(img)
    return img


def parse_model_data(model_path: Path):
    altitudes = parse_matrix_data(model_path / "altitude.dat").flatten()
    energies = parse_matrix_data(model_path / "energy.dat").flatten()
    matrix = parse_matrix_data(model_path / "M_emis.dat")
    return Model(
        altitudes=altitudes,
        energies=energies,
        matrix=matrix,
        # TODO: Load from file
        field_azimuth=185.8,
        field_elevation=77.4,
        box_min=(-50, -73, -80),
        box_max=(88, 65, 230),
    )

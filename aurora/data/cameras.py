from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, TYPE_CHECKING

from .common import PathLike

if TYPE_CHECKING:
    from aurora.camera import Camera


@dataclass
class CameraInfo:
    longitude: float
    latitude: float
    altitude: float
    location_name: str
    wavelength: Optional[str] = None


def load_cameras(
        cams_dir: PathLike,
        wl_filter: Optional[List[str]] = None,
        ds_factor: Optional[int] = None
    ) -> List["Camera"]:
    """
    Loads cameras from a position description file and image data directory.

    Args:
        imgs_dir: Path to the directory containing a camera_position.set file and
            camera image data. The directory structure is assumed to be either
            imgs_dir/{location}/*.dat for single wavelength or
            imgs_dir/{location}/{wavelength}/*.dat for multiple wavelengths.
        wl_filter (optional): List restricting the camera wavelengths to be loaded.
            If None, all available camera data is loaded.
        ds_factor (optional): Downsampling factor applied to camera calibration maps
            and loaded images.
    """
    cams_dir = Path(cams_dir)
    if not cams_dir.exists():
        raise FileNotFoundError(f"Camera directory {cams_dir} not found.")

    set_path = cams_dir / "camera_position.set"
    infos = load_camera_positions(set_path)
    if wl_filter is not None:
        infos = filter(lambda c: c.wavelength in wl_filter, infos)

    cameras = []
    for info in infos:
        cam_dir = cams_dir / info.location_name
        if info.wavelength is not None:
            cam_dir = cam_dir / info.wavelength
        if not cam_dir.exists():
            print(f"Warning: couldn't find images for camera {info.camera_id}.")
            continue
        cameras.append(load_camera(info, cam_dir, ds_factor=ds_factor))
    return cameras


def load_camera(
        cam_info: CameraInfo,
        cam_dir: PathLike,
        ds_factor: Optional[int] = None
    ) -> "Camera":
    """
    Load a camera object from camera metadata and a folder containing
    `az_cam.dat`, `ze_cam.dat`, and files matching `*image.dat`.
    """
    from aurora.camera import Camera

    cam_dir = Path(cam_dir)
    return Camera(
        latitude=cam_info.latitude,
        longitude=cam_info.longitude,
        altitude=cam_info.altitude,
        data_path=cam_dir,
        location=cam_info.location_name,
        wavelength=cam_info.wavelength,
        ds_factor=ds_factor,
    )


def load_camera_positions(set_path: PathLike) -> List[CameraInfo]:
    """
    Load camera positions and information from a .set file.
    """
    with open(set_path, "r") as f:
        data = f.read()

    sections = data.strip().split('----------------------------------')
    sections = [s.strip() for s in sections if s.strip()]
    cameras = []

    for idx, section in enumerate(sections, start=1):
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if len(lines) < 4:
            continue

        location_name = None
        wavelength = None
        coords = None

        i = 0
        while i < len(lines):
            line = lines[i]
            if 'position name' in line.lower():
                if i + 1 < len(lines):
                    location_name = lines[i + 1].strip()
                    i += 2
                    continue
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
            if 'wavelength' in line.lower():
                if i + 1 < len(lines):
                    try:
                        wavelength = lines[i + 1].strip()
                    except (ValueError, IndexError) as e:
                        raise ValueError(f"Error parsing wavelength in section {idx}: {e}")
                    i += 2
                    continue
            i += 1

        if coords is None:
            raise ValueError(f"Section {idx} missing coordinates")
        if location_name is None:
            raise ValueError(f"Section {idx} missing location name")

        cameras.append(
            CameraInfo(
                longitude=coords[0],
                latitude=coords[1],
                altitude=coords[2],
                location_name=location_name,
                wavelength=wavelength,
            )
        )

    return cameras


from typing import Optional, List
from pathlib import Path
from datetime import datetime
import re

import torch

from aurora.utils import PathLike, downsample_image
import aurora.geodesy as geo
import aurora.data as dat


class Image:
    def __init__(self, camera: 'Camera', path: Path, ds_factor: Optional[int] = None):
        self.camera = camera
        self.path = path
        self.date_time = self._parse_date_time()
        self.ds_factor = ds_factor
    
    def _parse_date_time(self):
        re_match = re.match(r'^(\d{8})_(\d{6})_image\.dat$', str(self.path.name))
        if re_match:
            date_str, time_str = re_match.group(1), re_match.group(2)
            try:
                dt = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
                return dt
            except ValueError as e:
                print(f"'{self.path}' has an invalid date/time: {e}")
        else:
            return None

    def load_image(self):
        img = dat.load_matrix_data(self.path)
        if self.ds_factor is not None:
            img = downsample_image(img, self.ds_factor)
        return img

    def downsample(self, factor: int, camera: Optional['Camera'] = None):
        ds_img = object.__new__(Image)
        ds_img.camera = self.camera if camera is None else camera
        ds_img.path = self.path
        ds_img.date_time = self.date_time
        ds_img.ds_factor = factor
        if self.ds_factor is not None:
            ds_img.ds_factor *= self.ds_factor
        return ds_img

class Camera:
    def __init__(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
        data_path: PathLike,
        location: Optional[str] = None,
        wavelength: Optional[str] = None,
        ds_factor: Optional[int] = None
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.location = location
        self.wavelength = wavelength
        self.data_path = Path(data_path)
        self.ds_factor = ds_factor

        self.azimuth = dat.load_matrix_data(self.data_path / "az_cam.dat")
        self.zenith = dat.load_matrix_data(self.data_path / "ze_cam.dat")
        if self.ds_factor is not None:
            self.azimuth = downsample_image(self.azimuth, self.ds_factor)
            self.zenith = downsample_image(self.zenith, self.ds_factor)
        
        self.images: List[Image] = []
        for img_file in sorted(self.data_path.rglob("*image.dat")):
            self.images.append(Image(self, img_file, self.ds_factor))
        
        if len(self.images) == 0:
            raise FileNotFoundError(f"No image file found in {self.data_path}.")

        self._image = None

    @property
    def width(self):
        return self.azimuth.size(1)
    
    @property
    def height(self):
        return self.azimuth.size(0)
    
    @property
    def image(self):
        if self._image is None:
            self._image = self.images[0].load_image()
        return self._image

    def downsample(self, factor: int):
        ds_cam = object.__new__(Camera)
        ds_cam.latitude = self.latitude
        ds_cam.longitude = self.longitude
        ds_cam.altitude = self.altitude
        ds_cam.location = self.location
        ds_cam.wavelength = self.wavelength
        ds_cam.data_path = self.data_path
        ds_cam.ds_factor = factor
        ds_cam.azimuth = downsample_image(self.azimuth, factor)
        ds_cam.zenith = downsample_image(self.zenith, factor)
        if self.ds_factor is not None:
            ds_cam.ds_factor *= self.ds_factor
        ds_cam.images = [img.downsample(factor, camera=ds_cam) for img in self.images]
        return ds_cam
    
    def create_rays_ecef(self, device=torch.device('cpu')):
        lat, lon = self.latitude, self.longitude
        alt = self.altitude
        
        az = self.azimuth.flatten().to(device)
        ze = self.zenith.flatten().to(device)
        rd_enu = geo.az_ze_to_enu(az, ze)
        rd_ecef = geo.rotate_enu_to_ecef(rd_enu, lat, lon)

        ro_ecef = geo.geodetic_to_ecef(lat, lon, alt).to(device)
        ro_ecef = ro_ecef.expand(rd_ecef.shape[0], -1)

        return ro_ecef, rd_ecef

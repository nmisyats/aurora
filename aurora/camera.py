from dataclasses import dataclass
from typing import Optional

import torch

from aurora.utils import downsample_image
import aurora.geodesy as geo


@dataclass
class Camera:
    camera_id: int
    name: str
    longitude: float
    latitude: float
    altitude: float
    location_name: str
    image: torch.Tensor
    azimuth: torch.Tensor
    zenith: torch.Tensor
    wavelength: Optional[str] = None

    @property
    def width(self):
        return self.azimuth.size(1)
    
    @property
    def height(self):
        return self.azimuth.size(0)

    def __repr__(self):
        return ("Camera=("
            f"camera_id={self.camera_id}, "
            f"name={self.name}, "
            f"location_name={self.name}, "
            f"latitude={self.latitude}, "
            f"longitude={self.longitude}, "
            f"altitude={self.altitude}, "
            f"image={type(self.image)} {tuple(self.image.shape)}, "
            f"azimuth={type(self.azimuth)} {tuple(self.azimuth.shape)}, "
            f"zenith={type(self.zenith)} {tuple(self.zenith.shape)}, "
            f"wavelength={self.wavelength}"
        ")")

    def downsample(self, factor: int):
        return Camera(
            camera_id=self.camera_id,
            name=self.name,
            longitude=self.longitude,
            latitude=self.latitude,
            altitude=self.altitude,
            location_name=self.location_name,
            image=downsample_image(self.image, factor),
            azimuth=downsample_image(self.azimuth, factor),
            zenith=downsample_image(self.zenith, factor),
            wavelength=self.wavelength
        )
    
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
from dataclasses import dataclass

import torch

from aurora.utils import downsample_image
import aurora.geodesy as geo


@dataclass
class Camera:
    name: str
    latitude: float
    longitude: float
    altitude: float
    image: torch.Tensor
    azimuth: torch.Tensor
    zenith: torch.Tensor

    @property
    def width(self):
        return self.azimuth.size(1)
    
    @property
    def height(self):
        return self.azimuth.size(0)

    def __repr__(self):
        return ", ".join(["Camera=(",
            f"name={self.name}",
            f"latitude={self.latitude}",
            f"longitude={self.longitude}",
            f"altitude={self.altitude}",
            f"image={type(self.image)} {self.image.shape}",
            f"azimuth={type(self.azimuth)} {self.azimuth.shape}",
            f"zenith={type(self.zenith)} {self.zenith.shape}",
        ")"])
    
    def __str__(self):
        return self.__repr__(self)

    def downsample(self, factor: int) -> 'Camera':
        return Camera(
            name=self.name,
            longitude=self.longitude,
            latitude=self.latitude,
            altitude=self.altitude,
            image=downsample_image(self.image, factor),
            azimuth=downsample_image(self.azimuth, factor),
            zenith=downsample_image(self.zenith, factor)
        )
    
    def create_rays_ecef(self, device=torch.device('cpu')):
        lat, lon = self.latitude, self.longitude
        alt = self.altitude

        az = self.azimuth.flatten().to(device)
        ze = self.zenith.flatten().to(device)
        rd_enu = geo.az_ze_to_enu(az, ze).to(device)
        rd_ecef = geo.rotate_enu_to_ecef(rd_enu, lat, lon)

        ro_ecef = geo.geodetic_to_ecef(lat, lon, alt).to(device)
        ro_ecef = ro_ecef.repeat(rd_ecef.shape[0], 1)

        return ro_ecef, rd_ecef
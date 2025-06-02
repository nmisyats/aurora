from dataclasses import dataclass

import torch

from aurora.utils import downsample_image
from aurora.frame import ReferenceFrame
from aurora.geodesy import (
    earth_radius,
    az_ze_to_UNE,
    lat_lon_to_ECEF
)


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
    
    def create_rays(self, frame: ReferenceFrame):
        az = self.azimuth.flatten().to(frame.device)
        ze = self.zenith.flatten().to(frame.device)
        rd_une = az_ze_to_UNE(az, ze).to(frame.device)
        rd_rel = frame.from_une(rd_une, is_point=False)

        lat, lon, alt = self.latitude, self.longitude, self.altitude
        ro_ecef_unit = lat_lon_to_ECEF(lat, lon).to(frame.device)
        radius = earth_radius(lat, lon) + alt
        ro_ecef = radius * ro_ecef_unit
        ro_rel = frame.from_ecef(ro_ecef, is_point=True)
        ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

        return ro_rel, rd_rel
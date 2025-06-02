from dataclasses import dataclass

import torch

from aurora.physics import ReferenceFrame
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    earth_radius,
)
from aurora.utils import downsample_image

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
    
    def create_rays(self, frame: ReferenceFrame, device: torch.device):
        lat, lon, alt = self.latitude, self.longitude, self.altitude
        o_ecef = frame.origin_ecef
        ecef_to_field = frame.ecef_to_field_mat

        az = self.azimuth.flatten().to(device)
        ze = self.zenith.flatten().to(device)
        rd_une = az_ze_to_UNE(az, ze).to(device)
        rd_ecef = UNE_to_ECEF(rd_une, lat, lon).to(device)
        rd_rel = torch.matmul(rd_ecef, ecef_to_field.T)

        ro_ecef_unit = lat_lon_to_ECEF(lat, lon).to(device)
        radius = earth_radius(lat, lon) + alt
        ro_ecef = radius * ro_ecef_unit
        ro_ecef_rel = ro_ecef - o_ecef
        ro_rel = torch.matmul(ro_ecef_rel, ecef_to_field.T)
        ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

        return ro_rel, rd_rel

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
from dataclasses import dataclass

import torch

from aurora.utils import downsample_image
import aurora.geodesy as geod


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
    
    # def create_rays_une(self, frame: MFAlignedFrame):
    #     az = self.azimuth.flatten().to(frame.device)
    #     ze = self.zenith.flatten().to(frame.device)
    #     rd_une = az_ze_to_UNE(az, ze).to(frame.device)
    #     rd = frame.from_UNE(rd_une, is_point=False)

    #     lat, lon, alt = self.latitude, self.longitude, self.altitude
    #     ro_ecef_unit = lat_lon_to_ECEF(lat, lon).to(frame.device)
    #     radius = earth_radius(lat, lon) + alt
    #     ro_ecef = radius * ro_ecef_unit
    #     ro = frame.from_ECEF(ro_ecef, is_point=True)
    #     ro = ro.repeat(rd.shape[0], 1)

    #     return ro, rd

    def create_rays_local_une(self, device=torch.device('cpu')):
        lat, lon = self.latitude, self.longitude
        alt = self.altitude

        az = self.azimuth.flatten().to(device)
        ze = self.zenith.flatten().to(device)
        rd_une = geod.az_ze_to_UNE(az, ze).to(device)

        ro_ecef_unit = geod.lat_lon_to_ECEF(lat, lon).to(device)
        ro_une_unit = geod.rotate_ECEF_to_UNE(ro_ecef_unit, lat, lon)
        ro_une = alt * ro_une_unit
        ro_une = ro_une.repeat(rd_une.shape[0], 1)

        return ro_une, rd_une
    
    def create_rays_ecef(self, device=torch.device('cpu')):
        lat, lon = self.latitude, self.longitude
        alt = self.altitude

        az = self.azimuth.flatten().to(device)
        ze = self.zenith.flatten().to(device)
        rd_une = geod.az_ze_to_UNE(az, ze).to(device)
        rd_ecef = geod.rotate_UNE_to_ECEF(rd_une, lat, lon)

        ro_ecef_unit = geod.lat_lon_to_ECEF(lat, lon).to(device)
        radius = geod.earth_radius(lat, lon) + alt
        ro_ecef = radius * ro_ecef_unit
        ro_ecef = ro_ecef.repeat(rd_ecef.shape[0], 1)

        return ro_ecef, rd_ecef
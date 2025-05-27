import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class Camera:
    name: str
    latitude: float
    longitude: float
    altitude: float
    image: torch.Tensor
    azimuth: torch.Tensor
    zenith: torch.Tensor

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

def downsample_image(image: torch.Tensor, factor: int) -> np.ndarray:
    """
    Downsamples a 2D or 3D image (e.g., grayscale or RGB) by picking every `factor`-th pixel.
    
    Parameters:
        image (np.ndarray): Input image matrix. Can be 2D (grayscale) or 3D (RGB).
        factor (int): Downsampling factor. Must be >= 1.
        
    Returns:
        np.ndarray: Downsampled image.
    """
    if factor < 1:
        raise ValueError("Downsampling factor must be >= 1")
    
    # Handle 2D (grayscale) or 3D (color) images
    if image.ndim == 2:
        return image[::factor, ::factor]
    else:
        return image[::factor, ::factor, :]

def downsample_camera(camera: Camera, factor: int) -> Camera:
    """
    Downsamples the image, azimuth, and zenith of a Camera object by a given factor."""
    return Camera(
        name=camera.name,
        longitude=camera.longitude,
        latitude=camera.latitude,
        altitude=camera.altitude,
        image=downsample_image(camera.image, factor),
        azimuth=downsample_image(camera.azimuth, factor),
        zenith=downsample_image(camera.zenith, factor)
    )
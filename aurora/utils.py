import numpy as np

from aurora.data import Camera

def downsample_image(image: np.ndarray, factor: int) -> np.ndarray:
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
    return Camera(
        name=camera.name,
        longitude=camera.longitude,
        latitude=camera.latitude,
        altitude=camera.altitude,
        image=downsample_image(camera.image, factor),
        azimuth=downsample_image(camera.azimuth, factor),
        zenith=downsample_image(camera.zenith, factor)
    )
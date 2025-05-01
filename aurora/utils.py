import numpy as np
from functools import wraps
import torch

from aurora.data import Camera

def numpify(func=None, *, device="cpu"):
    if func is None:
        return lambda f: numpify(f, device=device)

    @wraps(func)
    def wrapper(*args, **kwargs):
        def to_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            return x

        def to_numpy(x):
            if torch.is_tensor(x):
                return x.detach().cpu().numpy()
            elif isinstance(x, (list, tuple)):
                return type(x)(to_numpy(i) for i in x)
            elif isinstance(x, dict):
                return {k: to_numpy(v) for k, v in x.items()}
            return x

        args_t = tuple(to_tensor(a) for a in args)
        kwargs_t = {k: to_tensor(v) for k, v in kwargs.items()}
        result = func(*args_t, **kwargs_t)
        return to_numpy(result)

    return wrapper

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
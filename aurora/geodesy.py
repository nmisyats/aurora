import torch
import numpy as np

from aurora.utils import TensorLike, as_tensor

def az_ze_to_enu(azimuth: TensorLike, zenith: TensorLike):
    """
    Convert azimuth and zenith angles to East-North-Up (ENU) coordinate system vectors.
    
    Args:
        azimuth: Azimuth angle in degrees measured clockwise from North (0° at North, 
            90° at East, 180° at South, 270° at West)
        zenith : Zenith angle in degrees measured from the vertical (0° at zenith/up, 
            90° at horizon, 180° at nadir/down)
    
    Returns:
        torch.Tensor: A tensor of shape (*azimuth.shape, 3) where the components of the last dimension
            represent the ENU vector for the corresponding azimuth/zenith values:
            - east: Component pointing toward geographic East
            - north: Component pointing toward geographic North
            - up: Vertical component pointing away from Earth's center
            
    Note:
        The returned vector components form a unit vector (magnitude = 1)
        when the input angles represent a direction in 3D space.
    """
    azimuth = as_tensor(azimuth)
    zenith = as_tensor(zenith)
    
    az_rad = torch.deg2rad(azimuth)
    ze_rad = torch.deg2rad(zenith)

    east = torch.sin(az_rad) * torch.sin(ze_rad)
    north = torch.cos(az_rad) * torch.sin(ze_rad)
    up = torch.cos(ze_rad)

    return torch.stack((east, north, up), dim=-1)

def rotate_enu_to_ecef(enu: TensorLike, lat: float, lon: float):
    """
    Convert East-North-Up coordinates to Earth-Centered, Earth-Fixed coordinates.
    
    Args:
        enu: Tensor of shape (*, 3) of coordinates in ENU frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        torch.Tensor: coordinates in ECEF frame, forming a unit vector, for each
            input ENU coordinates
    """
    enu = as_tensor(enu)

    # Extract ENU components
    east, north, up = enu[..., 0], enu[..., 1], enu[..., 2]
    
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Calculate sines and cosines
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Calculate the transformation from ENU to ECEF
    x = -sin_lon * east - sin_lat * cos_lon * north + cos_lat * cos_lon * up
    y = cos_lon * east - sin_lat * sin_lon * north + cos_lat * sin_lon * up
    z = cos_lat * north + sin_lat * up
    
    return torch.stack((x, y, z), dim=-1).to(torch.float32)

def rotate_ecef_to_enu(ecef: TensorLike, lat: float, lon: float):
    """
    Convert Earth-Centered, Earth-Fixed coordinates to East-North-Up coordinates.
    
    Args:
        ecef: Tensor of shape (*, 3) of coordinates in ECEF frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        torch.Tensor: coordinates in ENU frame, forming a unit vector, for each
            input ECEF coordinates
    """
    ecef = as_tensor(ecef)

    # Extract ECEF components
    x, y, z = ecef[..., 0], ecef[..., 1], ecef[..., 2]
    
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Calculate sines and cosines
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    
    # Calculate the transformation from ECEF to ENU (inverse of ENU to ECEF)
    east = -sin_lon * x + cos_lon * y
    north = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
    up = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z
    
    return torch.stack((east, north, up), dim=-1).to(torch.float32)

def geodetic_to_ecef(lat: TensorLike, lon: TensorLike, alt: TensorLike = 0.0):
    """
    Convert latitude and longitude (in degrees) to ECEF coordinates.
    
    Parameters:
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        alt: Altitude in kilometers (default is 0).
    
    Returns:
        torch.Tensor: coordinates in ECEF.
    """
    lat = as_tensor(lat)
    lon = as_tensor(lon)

    # Convert lat/lon to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    ecef_unit = torch.stack((x, y, z), dim=-1)

    r = earth_radius(lat, lon) + as_tensor(alt)

    ecef = ecef_unit * r.unsqueeze(-1)

    return ecef

def enu_basis_vectors_ecef(lat: float, lon: float):
    """
    Get the ENU basis vectors in ECEF coordinates for a given location.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        tuple: (east_dir, north_dir, up_dir) - ENU basis vectors in ECEF frame
    """
    east_dir  = rotate_enu_to_ecef(torch.tensor([1, 0, 0]), lat, lon)
    north_dir = rotate_enu_to_ecef(torch.tensor([0, 1, 0]), lat, lon)
    up_dir    = rotate_enu_to_ecef(torch.tensor([0, 0, 1]), lat, lon)
    
    return east_dir, north_dir, up_dir

def earth_radius(lat: TensorLike, lon: TensorLike):
    """
    Get the approximate Earth radius at a given latitude and longitude.
    """
    return torch.full_like(as_tensor(lat), 6371.0) # Approximate Earth radius in km

def inc_dec_to_enu(inclination: TensorLike, declination: TensorLike):
    """
    Convert inclination and declination angles to East-North-Up (ENU) coordinate system vectors.
    
    Args:
        inclination: Inclination angle in degrees (positive downward from horizontal)
        declination: Declination angle in degrees (positive eastward from north)
    
    Returns:
        torch.Tensor: A tensor of shape (*inclination.shape, 3) where the components of the last dimension
            represent the ENU vector for the corresponding inclination/declination values:
            - east: Component pointing toward geographic East
            - north: Component pointing toward geographic North  
            - up: Vertical component pointing away from Earth's center
    """
    inclination = as_tensor(inclination)
    declination = as_tensor(declination)

    # Convert angles from degrees to radians
    inc_rad = torch.deg2rad(inclination)
    dec_rad = torch.deg2rad(declination)
    
    east = torch.sin(dec_rad) * torch.cos(inc_rad)
    north = torch.cos(dec_rad) * torch.cos(inc_rad)
    up = -torch.sin(inc_rad)

    # Return the unit vector in ENU frame
    return torch.stack((east, north, up), dim=-1)
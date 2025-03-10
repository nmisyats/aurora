import torch
import numpy as np

def az_ze_to_UNE(azimuth: torch.Tensor, zenith: torch.Tensor):
    """
    Convert azimuth and zenith angles to Up-North-East (UNE) coordinate system vectors.
    
    Args:
        azimuth: Azimuth angle in degrees measured clockwise from North (0° at North, 
            90° at East, 180° at South, 270° at West)
        zenith : Zenith angle in degrees measured from the vertical (0° at zenith/up, 
            90° at horizon, 180° at nadir/down)
    
    Returns:
        torch.Tensor: A tensor of shape (*azimuth.shape, 3) where the coomponents of the last dimension
            represent the UNE vector for the corresponding azimuth/latitude values:
            - up: Vertical component pointing away from Earth's center
            - north: Component pointing toward geographic North
            - east: Component pointing toward geographic East
            
    Note:
        The returned vector components form a unit vector (magnitude = 1)
        when the input angles represent a direction in 3D space.
    """
    az_rad = torch.deg2rad(azimuth)
    ze_rad = torch.deg2rad(zenith)

    up = torch.cos(ze_rad)
    north = torch.cos(az_rad) * torch.sin(ze_rad)
    east = torch.sin(az_rad) * torch.sin(ze_rad)

    return torch.stack((up, north, east), dim=-1)

def UNE_to_ECEF(une: torch.Tensor, lat: float, lon: float):
    """
    Convert Up-North-East coordinates to Earth-Centered, Earth-Fixed coordinates.
    
    Args:
        une: Tensor of shape (*, 3) of coordinates in UNE frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        torch.Tensor: coordinates in ECEF frame, forming a unit vector, for each
            input UNE coordinates
    """
    # Extract UNE components
    up, north, east = une[..., 0], une[..., 1], une[..., 2]
    
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Calculate sines and cosines
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    # Calculate the transformation from UNE to ECEF
    x = -sin_lat * cos_lon * north - sin_lon * east + cos_lat * cos_lon * up
    y = -sin_lat * sin_lon * north + cos_lon * east + cos_lat * sin_lon * up
    z = cos_lat * north + sin_lat * up
    
    return torch.stack((x, y, z), dim=-1)

def ECEF_to_UNE(ecef: torch.Tensor, lat: float, lon: float):
    """
    Convert Earth-Centered, Earth-Fixed coordinates to Up-North-East coordinates.
    
    Args:
        ecef: Tensor of shape (*, 3) of coordinates in UNE frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        torch.Tensor: coordinates in UNE frame, forming a unit vector, for each
            input ECEF coordinates
    """
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
    
    # Calculate the transformation from ECEF to UNE (inverse of UNE to ECEF)
    up = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z
    north = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
    east = -sin_lon * x + cos_lon * y
    
    return torch.stack((up, north, east), dim=-1)

def lat_lon_to_ECEF(lat: float, lon: float):
    """
    Convert latitude and longitude (in degrees) to normalized ECEF coordinates (magnitude 1).
    
    Parameters:
        lat_deg: Latitude in degrees.
        lon_deg: Longitude in degrees.
    
    Returns:
        torch.Tensor: coordinates in ECEF (unit sphere).
    """
    # Convert lat/lon to radians
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return torch.tensor([x, y, z])

def UNE_basis_ECEF(lat: float, lon: float):
    up_dir    = UNE_to_ECEF(torch.tensor([1, 0, 0]), lat, lon)
    north_dir = UNE_to_ECEF(torch.tensor([0, 1, 0]), lat, lon)
    east_dir  = UNE_to_ECEF(torch.tensor([0, 0, 1]), lat, lon)
    
    return up_dir, north_dir, east_dir

def earth_radius(lat: float, lon: float):
    return 6371.0 # km
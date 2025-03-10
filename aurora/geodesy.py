import torch

def az_ze_to_UNE(azimuth, zenith):
    """
    Convert azimuth and zenith angles to Up-North-East (UNE) coordinate system vectors.
    
    Args:
        azimuth (torch.Tensor): Azimuth angle in degrees measured clockwise from North (0° at North, 
                                90° at East, 180° at South, 270° at West)
        zenith (torch.Tensor): Zenith angle in degrees measured from the vertical (0° at zenith/up, 
                               90° at horizon, 180° at nadir/down)
    
    Returns:
        tuple: A tuple containing three torch.Tensor components:
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
    
    return up, north, east

def UNE_to_ECEF(une, lat, lon):
    """
    Convert Up-North-East coordinates to Earth-Centered, Earth-Fixed coordinates.
    
    Args:
        une: Tuple of (up, north, east) coordinates in UNE frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        Tuple of (x, y, z) coordinates in ECEF frame, forming a unit vector
    """
    # Extract UNE components
    up, north, east = une
    
    # Convert lat/lon to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    
    # Calculate sines and cosines
    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)
    sin_lon = torch.sin(lon_rad)
    cos_lon = torch.cos(lon_rad)

    # Calculate the transformation from UNE to ECEF
    x = -sin_lat * cos_lon * north - sin_lon * east + cos_lat * cos_lon * up
    y = -sin_lat * sin_lon * north + cos_lon * east + cos_lat * sin_lon * up
    z = cos_lat * north + sin_lat * up
    
    return x, y, z

def ECEF_to_UNE(ecef, lat, lon):
    """
    Convert Earth-Centered, Earth-Fixed coordinates to Up-North-East coordinates.
    
    Args:
        ecef: Tuple of (x, y, z) coordinates in ECEF frame
        lat: Latitude in degrees
        lon: Longitude in degrees
        
    Returns:
        Tuple of (up, north, east) coordinates in UNE frame
    """
    # Extract ECEF components
    x, y, z = ecef
    
    # Convert lat/lon to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)
    
    # Calculate sines and cosines
    sin_lat = torch.sin(lat_rad)
    cos_lat = torch.cos(lat_rad)
    sin_lon = torch.sin(lon_rad)
    cos_lon = torch.cos(lon_rad)
    
    # Calculate the transformation from ECEF to UNE (inverse of UNE to ECEF)
    up = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z
    north = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
    east = -sin_lon * x + cos_lon * y
    
    return up, north, east

def lat_lon_to_ECEF(lat, lon):
    """
    Convert latitude and longitude (in degrees) to normalized ECEF coordinates (magnitude 1).
    
    Parameters:
        lat_deg (float): Latitude in degrees.
        lon_deg (float): Longitude in degrees.
    
    Returns:
        tuple: (x, y, z) coordinates in ECEF (unit sphere).
    """
    # Convert lat/lon to radians
    lat_rad = torch.deg2rad(lat)
    lon_rad = torch.deg2rad(lon)

    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)

    return x, y, z

def UNE_basis_ECEF(lat, lon):
    v0 = torch.zeros_like(lat)
    v1 = torch.ones_like(lat)

    u_dir = UNE_to_ECEF((v1, v0, v0), lat, lon)
    n_dir = UNE_to_ECEF((v0, v1, v0), lat, lon)
    e_dir = UNE_to_ECEF((v0, v0, v1), lat, lon)
    
    return u_dir, n_dir, e_dir

def earth_radius(lat, lon):
    return 6371.0
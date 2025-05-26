import torch

from aurora.camera import Camera
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE
)

def get_frame_transform(o_lat: float, o_lon: float, o_alt: float, field_inc: float, field_dec: float, device: torch.device):
    o_ecef_unit = lat_lon_to_ECEF(o_lat, o_lon)
    radius = earth_radius(o_lat, o_lon) + o_alt
    o_ecef = radius * o_ecef_unit
    o_ecef = o_ecef.to(device)

    une_to_ecef = torch.stack(UNE_basis_ECEF(o_lat, o_lon)).T
    ecef_to_une = torch.linalg.inv(une_to_ecef)
    field_dir_une = inc_dec_to_UNE(
        torch.scalar_tensor(field_inc),
        torch.scalar_tensor(field_dec)
    )
    field_to_une = torch.stack((
        torch.tensor([0.0, -1.0, 0.0]),
        torch.tensor([0.0,  0.0, 1.0]),
        -field_dir_une
    )).T
    une_to_field = torch.linalg.inv(field_to_une)
    ecef_to_field = torch.matmul(une_to_field, ecef_to_une)
    ecef_to_field = ecef_to_field.to(device)

    # metric tensor
    metric_tensor = torch.matmul(ecef_to_field, ecef_to_field.T)
    metric_tensor = metric_tensor.to(device)

    return o_ecef, ecef_to_field, metric_tensor

def create_camera_rays(cam: Camera, o_ecef: torch.Tensor, ecef_to_field: torch.Tensor, device: torch.device):
    lat, lon, alt = cam.latitude, cam.longitude, cam.altitude

    az = torch.from_numpy(cam.azimuth).flatten().to(device)
    ze = torch.from_numpy(cam.zenith).flatten().to(device)
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

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float, num_bins: int, device: torch.device):
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1, device=device) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins, device=device) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd

def ray_box_intersection(ro: torch.Tensor, rd: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor):
    """
    Compute the intersection distances t_n (near) and t_f (far) of multiple rays with an axis-aligned bounding box.
    
    Parameters:
        ro: (n, 3) tensor representing the ray origins.
        rd: (n, 3) tensor representing the ray directions.
        box_min: 1D tensor [x_min, y_min, z_min] representing the min coordinates of the box.
        box_max: 1D tensor [x_max, y_max, z_max] representing the max coordinates of the box.
    
    Returns:
        tuple: (t_n, t_f) where t_n and t_f are 1D tensors of shape (n,), containing intersection distances 
               for each ray or (None, None) if no valid intersections.
    """
    inv_dir = 1.0 / rd
    t_min = (box_min - ro) * inv_dir
    t_max = (box_max - ro) * inv_dir
    
    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)
    
    tn = torch.max(t0, dim=1).values
    tf = torch.min(t1, dim=1).values
    
    inside_mask = ((ro >= box_min) & (ro <= box_max)).all(dim=1)
    
    tf[inside_mask] = 0.0
    tn[inside_mask] = torch.max(t1[inside_mask], dim=1).values
    
    mask = (tn <= tf) & (tf >= 0)
    tn[~mask] = float('nan')
    tf[~mask] = float('nan')
    
    return tn, tf
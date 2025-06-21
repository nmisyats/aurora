import torch

def sample_ray_points(ro: torch.Tensor, rd: torch.Tensor, t: torch.Tensor):
    """
    Samples points along the rays for each given distance.
    
    Args:
        ro: (n, 3) tensor representing ray origins.
        rd: (n, 3) tensor representing ray directons.
        t: (n, n_samples) or (n_samples,) tensor representing each distances to sample each ray.
    
    Returns:
        Tensor of shape (n, num_samples, 3) of each sampled positions.
    """
    # ro: (n, 3)
    # rd: (n, 3)
    # t: (n, n_samples) or (n_samples,)
    if t.ndim == 1:
        t = t.expand(ro.shape[0], -1) # (n, n_samples)
    n, n_samples = t.shape
    ro = ro.view(n, 1, 3).expand(-1, n_samples, -1) # (n, n_samples, 3)
    rd = rd.view(n, 1, 3).expand(-1, n_samples, -1) # (n, n_samples, 3)
    t = t.unsqueeze(-1) # (n, n_samples, 1)
    return ro + t * rd # (n, n_samples, 3)


def inside_box_mask(p: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor):
    """
    Check if points are inside an axis-aligned bounding box.
    
    Args:
        p: (n, 3) tensor representing the points.
        box_min: 1D tensor [x_min, y_min, z_min] representing the min coordinates of the box.
        box_max: 1D tensor [x_max, y_max, z_max] representing the max coordinates of the box.
    
    Returns:
        Tensor of shape (n,) with boolean values indicating if each point is inside the box.
    """
    return ((p >= box_min) & (p <= box_max)).all(dim=-1)

def ray_box_intersection(
        ro: torch.Tensor,
        rd: torch.Tensor,
        box_min: torch.Tensor,
        box_max: torch.Tensor
    ):
    """
    Compute the intersection distances t_n (near) and t_f (far) of multiple rays with an
    axis-aligned bounding box.
    
    Args:
        ro: (n, 3) tensor representing the ray origins.
        rd: (n, 3) tensor representing the ray directions.
        box_min: 1D tensor [x_min, y_min, z_min] representing the min coordinates of the box.
        box_max: 1D tensor [x_max, y_max, z_max] representing the max coordinates of the box.
    
    Returns:
        tuple: (t_n, t_f) where t_n and t_f are 1D tensors of shape (n,), containing intersection
            distances for each ray or (None, None) if no valid intersections.
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

import torch

def create_ray_points(
        ro: torch.Tensor,
        rd: torch.Tensor,
        min_t: torch.Tensor,
        max_t: torch.Tensor,
        num_bins: int,
        random: bool
    ):
    """
    Create points along a ray defined by origin `ro` and direction `rd` within the range [min_t, max_t].
    The points are uniformly distributed in `num_bins` bins.
    Parameters:
        ro: (n, 3) tensor representing the ray origins.
        rd: (n, 3) tensor representing the ray directions.
        min_t: (n,) tensor representing distance along the ray.
        max_t: (n,) tensor representing maximum distance along the ray.
        num_bins: Number of bins to divide the range [min_t, max_t].
        random: Whether to sample points randomly in each bin or in the middle

    Returns:
        (t, p): t (n, num_bins) tensor of uniformly distributed distances along the ray
        and p (n, num_bins, 3) tensor of points along the ray.
    """
    device = ro.device
    n = ro.shape[0]

    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device) # num_bins + 1 edges (num_bins + 1,)
    bin_edges = bin_edges.expand(n, -1) # (n, num_bins + 1)
    min_t = min_t.unsqueeze(-1) # (n, 1)
    max_t = max_t.unsqueeze(-1) # (n, 1)
    bin_edges = bin_edges * (max_t - min_t) + min_t
    
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:, :-1] # (n, num_bins)
    upper_edges = bin_edges[:, 1:] # (n, num_bins)

    if random:
        # Generate random values in each bin
        t01 = torch.rand(n, num_bins, device=device) # (n, num_bins)
        t = lower_edges + t01 * (upper_edges - lower_edges) # (n, num_bins)
    else:
        # Generate midpoints in each bin
        t = 0.5 * (lower_edges + upper_edges) # (n, num_bins)
    # Create points
    # (n, 3) -> (n, 1, 3) -> (n, num_bins, 3)
    ro = ro.unsqueeze(1).expand(-1, num_bins, -1)
    rd = rd.unsqueeze(1).expand(-1, num_bins, -1)
    p = ro + t.unsqueeze(-1) * rd # t: (n, num_bins, 1)
    return t, p

def inside_box_mask(p: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor):
    """
    Check if points are inside an axis-aligned bounding box.
    
    Parameters:
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
    
    Parameters:
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

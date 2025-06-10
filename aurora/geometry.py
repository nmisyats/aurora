import torch

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float, num_bins: int, device: torch.device):
    """
    Create points along a ray defined by origin `ro` and direction `rd` within the range [min_t, max_t].
    The points are uniformly distributed in `num_bins` bins.
    Parameters:
        ro: (n, 3) tensor representing the ray origins.
        rd: (n, 3) tensor representing the ray directions.
        min_t: Minimum distance along the ray.
        max_t: Maximum distance along the ray.
        num_bins: Number of bins to divide the range [min_t, max_t].
        device: Device to create tensors on.

    Returns:
        t: (num_bins,) tensor of uniformly distributed distances along the ray.
        p: (num_bins, 3) tensor of points along the ray.
    """
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1, device=device) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins, device=device) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd

def inside_box_mask(p: torch.Tensor, box_min: torch.Tensor, box_max: torch.Tensor):
    """
    Check if points are inside an axis-aligned bounding box.
    
    Parameters:
        p: (*, 3) tensor representing the points.
        box_min: 1D tensor [x_min, y_min, z_min] representing the min coordinates of the box.
        box_max: 1D tensor [x_max, y_max, z_max] representing the max coordinates of the box.
    
    Returns:
        Tensor of shape (n,) with boolean values indicating if each point is inside the box.
    """
    return ((p >= box_min) & (p <= box_max)).all(dim=-1)

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

class BBox:
    def __init__(self, box_min: torch.Tensor, box_max: torch.Tensor):
        """
        Initialize an axis-aligned bounding box with minimum and maximum coordinates.
        
        Parameters:
            box_min: 1D tensor [x_min, y_min, z_min] representing the min coordinates of the box.
            box_max: 1D tensor [x_max, y_max, z_max] representing the max coordinates of the box.
        """
        self.box_min = box_min
        self.box_max = box_max

    def __repr__(self):
        return f"BBbox(min={self.box_min.tolist()}, max={self.box_max.tolist()})"
    
    def contains(self, p: torch.Tensor):
        """
        Check if points are inside the bounding box.
        
        Parameters:
            p: (*, 3) tensor representing the points.
        
        Returns:
            Tensor of shape (n,) with boolean values indicating if each point is inside the box.
        """
        return inside_box_mask(p, self.box_min, self.box_max)
    
    def intersection(self, ro: torch.Tensor, rd: torch.Tensor):
        """
        Compute the intersection distances t_n (near) and t_f (far) of rays with the bounding box.
        
        Parameters:
            ro: (n, 3) tensor representing the ray origins.
            rd: (n, 3) tensor representing the ray directions.
        
        Returns:
            tuple: (t_n, t_f) where t_n and t_f are 1D tensors of shape (n,), containing
                intersection distances for each ray or (None, None) if no valid intersections.
        """
        return ray_box_intersection(ro, rd, self.box_min, self.box_max)
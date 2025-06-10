import torch

from aurora.frame import Frame
import aurora.geometry as geom


class BBox:
    def __init__(
            self,
            frame: Frame,
            south_range: tuple[float, float],
            east_range: tuple[float, float],
            altitude_range: tuple[float, float]
        ):
        """
        Initialize an oblique bounding box in the given frame.

        Parameters:
            frame: Frame object defining the reference frame.
            north_south_range: Tuple (min, max) for north-south coordinates in meters.
            west_east_range: Tuple (min, max) for west-east coordinates in meters.
            altitude_range: Tuple (min, max) for altitude in meters.
        """
        x_min, x_max = south_range
        y_min, y_max = east_range
        h_min, h_max = altitude_range

        # Shift and scale z-axis to account for oblicity and altitude
        o_alt = frame.origin_altitude.item()
        z_met = frame.metric_tensor[2,2].item()
        z_min = (h_min - o_alt) / z_met
        z_max = (h_max - o_alt) / z_met

        self.xyz_min = torch.tensor([x_min, y_min, z_min], device=frame.device)
        self.xyz_max = torch.tensor([x_max, y_max, z_max], device=frame.device)

    def __repr__(self):
        return f"BBbox(min={self.xyz_min.tolist()}, max={self.xyz_max.tolist()})"
    
    def contains(self, p: torch.Tensor):
        """
        Check if points are inside the bounding box.
        
        Parameters:
            p: (*, 3) tensor representing the points.
        
        Returns:
            Tensor of shape (n,) with boolean values indicating if each point is inside the box.
        """
        return geom.inside_box_mask(p, self.xyz_min, self.xyz_max)
    
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
        return geom.ray_box_intersection(ro, rd, self.xyz_min, self.xyz_max)
    
    def __repr__(self):
        return f"BBox(xyz_min={self.xyz_min.tolist()}, xyz_max={self.xyz_max.tolist()})"
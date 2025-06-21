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
        o_alt = frame.origin_altitude
        z_met = frame.metric_tensor[2,2].item()
        z_min = (h_min - o_alt) / z_met
        z_max = (h_max - o_alt) / z_met

        self.xyz_min = torch.tensor([x_min, y_min, z_min], device=frame.device)
        self.xyz_max = torch.tensor([x_max, y_max, z_max], device=frame.device)

        self.south_range = south_range
        self.east_range = east_range
        self.altitude_range = altitude_range
    
    @property
    def device(self):
        return self.xyz_min.device
    
    @property
    def xy_min(self):
        return self.xyz_min[:2]
    
    @property
    def xy_max(self):
        return self.xyz_max[:2]
    
    def contains(self, p: torch.Tensor):
        return geom.inside_box_mask(p, self.xyz_min, self.xyz_max)
    
    def intersection(self, ro: torch.Tensor, rd: torch.Tensor):
        return geom.ray_box_intersection(ro, rd, self.xyz_min, self.xyz_max)

    def __repr__(self):
        return (
            f"BBbox("
            f"xyz_min={self.xyz_min.tolist()}, "
            f"xyz_max={self.xyz_max.tolist()}"
            f")"
        )
    
    def to(self, device):
        """Move all tensors to the specified device and return a new BBox instance."""
        new_bbox = object.__new__(BBox)

        new_bbox.xyz_min = self.xyz_min.to(device)
        new_bbox.xyz_max = self.xyz_max.to(device)

        new_bbox.south_range = self.south_range
        new_bbox.east_range = self.east_range
        new_bbox.altitude_range = self.altitude_range
        
        return new_bbox


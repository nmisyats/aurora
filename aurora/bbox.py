from typing import Tuple

import torch

from aurora.frame import Frame
import aurora.geometry as gmt
from aurora.utils import xy_grid, xyz_grid


class BBox:
    def __init__(
            self,
            frame: Frame,
            south_range: Tuple[float, float],
            east_range: Tuple[float, float],
            altitude_range: Tuple[float, float]
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
        z_min = frame.altitude_to_z(h_min)
        z_max = frame.altitude_to_z(h_max)

        self.xyz_min = torch.tensor([x_min, y_min, z_min], device=frame.device)
        self.xyz_max = torch.tensor([x_max, y_max, z_max], device=frame.device)

        self.south_range = south_range
        self.east_range = east_range
        self.altitude_range = altitude_range
    
    @property
    def device(self):
        return self.xyz_min.device
    
    def norm_xyz(self, xyz: torch.Tensor):
        return (xyz - self.xyz_min) / (self.xy_max - self.xyz_min)
    
    def real_xyz(self, xyz_norm: torch.Tensor):
        return self.xyz_min + xyz_norm * (self.xyz_max - self.xyz_min)
    
    def norm_xy(self, xy: torch.Tensor):
        return (xy - self.xy_min) / (self.xy_max - self.xy_min)
    
    def real_xy(self, xy_norm: torch.Tensor):
        return self.xy_min + xy_norm * (self.xy_max - self.xy_min)
    
    def contains(self, p: torch.Tensor):
        return gmt.inside_box_mask(p, self.xyz_min, self.xyz_max)
    
    def intersection(self, ro: torch.Tensor, rd: torch.Tensor):
        return gmt.ray_box_intersection(ro, rd, self.xyz_min, self.xyz_max)
    
    def xy_grid(self, res_x: int, res_y: int):
        return xy_grid(self.xy_min, self.xy_max, res_x, res_y)

    def xyz_grid(self, res_x: int, res_y: int, res_z: int):
        return xyz_grid(self.xyz_min, self.xyz_max, res_x, res_y, res_z)
    
    @property
    def xy_min(self):
        return self.xyz_min[:2]
    
    @property
    def xy_max(self):
        return self.xyz_max[:2]
    
    @property
    def xy_bounds(self):
        return (self.xy_min, self.xy_max)
    
    @property
    def xyz_bounds(self):
        return (self.xyz_min, self.xyz_max)
    
    @property
    def x_min(self) -> float:
        return self.xyz_min[0].item()
    
    @property
    def x_max(self) -> float:
        return self.xyz_max[0].item()
    
    @property
    def y_min(self) -> float:
        return self.xyz_min[1].item()
    
    @property
    def y_max(self) -> float:
        return self.xyz_max[1].item()
    
    @property
    def z_min(self) -> float:
        return self.xyz_min[2].item()
    
    @property
    def z_max(self) -> float:
        return self.xyz_max[2].item()

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


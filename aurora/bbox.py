import torch

from aurora.frame import MFAlignedFrame
import aurora.geometry as geom


class MFAlignedBBox:
    def __init__(
            self,
            north_range: tuple[float, float],
            east_range: tuple[float, float],
            height: float,
            frame: MFAlignedFrame
        ):
        self.frame = frame
        self.device = frame.device

        north_min, north_max = north_range
        east_min, east_max = east_range
        # Scale upmin to maintain oblicity
        height = height / frame.metric_tensor[2,2]

        une_min = torch.tensor([  0.0,  north_min, east_min], device=self.device)
        une_max = torch.tensor([height, north_max, east_max], device=self.device)

        self.xyz_min = frame.from_local_UNE(une_min)
        self.xyz_max = frame.from_local_UNE(une_max)
        self.xy_min = self.xyz_min[:2]
        self.xy_max = self.xyz_max[:2]
        
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
    
    def __getitem__(self, idx):
        return self.xyz_min[idx], self.xyz_max[idx]
    
    def __str__(self):
        return ("BBox("
            f"x=({self.x_min:.2f}, {self.x_max:.2f}),"
            f"y=({self.y_min:.2f}, {self.y_max:.2f}),"
            f"z=({self.z_min:.2f}, {self.z_max:.2f}),"
            f"device={self.device}"
            ")")
    
    def ray_intersect_frame(self, ro_frame: torch.Tensor, rd_frame: torch.Tensor):
        tn_frame, tf_frame = geom.ray_box_intersection(
            ro_frame, rd_frame, self.xyz_min, self.xyz_max)
        return tn_frame, tf_frame

    def ray_intersect_local_une(self, ro_une: torch.Tensor, rd_une: torch.Tensor):
        ro_frame = self.frame.from_local_UNE(ro_une, is_point=True)
        rd_frame = self.frame.from_local_UNE(rd_une, is_point=False)
        tn_frame, tf_frame = self.ray_intersect_frame(ro_frame, rd_frame)
        scale = self.frame.metric_scale(rd_frame)
        tn_une = tn_frame * scale
        tf_une = tf_frame * scale
        return tn_une, tf_une
    
    def ray_intersect_ecef(self, ro_ecef: torch.Tensor, rd_ecef: torch.Tensor):
        ro_frame = self.frame.from_ECEF(ro_ecef, is_point=True)
        rd_frame = self.frame.from_ECEF(rd_ecef, is_point=False)
        tn_frame, tf_frame = self.ray_intersect_frame(ro_frame, rd_frame)
        scale = self.frame.metric_scale(rd_frame)
        tn_ecef = tn_frame * scale
        tf_ecef = tf_frame * scale
        return tn_ecef, tf_ecef
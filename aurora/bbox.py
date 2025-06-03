import torch

from aurora.frame import ReferenceFrame

class BBox:
    def __init__(
            self,
            up_range: tuple[float, float],
            north_range: tuple[float, float],
            east_range: tuple[float, float],
            frame: ReferenceFrame,
            device: torch.device
    ):
        self.device = device

        up_min, up_max = up_range
        north_min, north_max = north_range
        east_min, east_max = east_range
        une_min = torch.tensor([up_min, north_min, east_min], device=device)
        une_max = torch.tensor([up_max, north_max, east_max], device=device)

        self.xyz_min = frame.from_une(une_min)
        self.xyz_max = frame.from_une(une_max)
        self.xy_min = self.xyz_max[:2]
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
        return "BBox(" + ", ".join([
            f"x=({self.x_min:.2f}, {self.x_max:.2f})",
            f"y=({self.y_min:.2f}, {self.y_max:.2f})",
            f"z=({self.z_min:.2f}, {self.z_max:.2f})",
            f"device={self.device}"
        ]) + ")"
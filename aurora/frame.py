import torch

from aurora.geodesy import (
    lat_lon_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE,
)


class MFAlignedFrame:
    def __init__(self,
        origin_latitude: float,
        origin_longitude: float,
        origin_altitude: float,
        field_inclination: float,
        field_declination: float,
        device: torch.device
    ):
        self.device = device
        
        # Get the frame transforms
        o_lat, o_lon, o_alt = origin_latitude, origin_longitude, origin_altitude
        o_ecef_unit = lat_lon_to_ECEF(o_lat, o_lon)
        radius = earth_radius(o_lat, o_lon) + o_alt
        o_ecef = radius * o_ecef_unit
        o_ecef = o_ecef.to(device)

        une_to_ecef = torch.stack(UNE_basis_ECEF(o_lat, o_lon)).T
        ecef_to_une = torch.linalg.inv(une_to_ecef)
        field_dir_une = inc_dec_to_UNE(field_inclination, field_declination)
        field_to_une = torch.stack((
            torch.tensor([0.0, -1.0, 0.0]),
            torch.tensor([0.0,  0.0, 1.0]),
            -field_dir_une
        )).T
        une_to_field = torch.linalg.inv(field_to_une)
        ecef_to_field = torch.matmul(une_to_field, ecef_to_une)
        field_to_ecef = torch.matmul(une_to_ecef, field_to_une)
        
        field_to_une = field_to_une.to(device)
        une_to_field = une_to_field.to(device)
        ecef_to_field = ecef_to_field.to(device)
        field_to_ecef = field_to_ecef.to(device)

        metric_tensor = torch.matmul(ecef_to_field, ecef_to_field.T)
        metric_tensor = metric_tensor.to(device)

        # Store the reference frame parameters
        self.origin_ecef = o_ecef
        self.une_to_field_mat = une_to_field
        self.field_to_une_mat = field_to_une
        self.ecef_to_field_mat = ecef_to_field
        self.field_to_ecef_mat = field_to_ecef
        self.metric_tensor = metric_tensor

        self.origin_latitude = torch.scalar_tensor(o_lat, device=device)
        self.origin_longitude = torch.scalar_tensor(o_lon, device=device)
        self.origin_altitude = torch.scalar_tensor(o_alt, device=device)

    def metric_scale(self, d_frame: torch.Tensor) -> torch.Tensor:
        d, m = d_frame, self.metric_tensor
        if d.ndim == 1:
            return torch.sqrt(d.T @ m @ d)
        elif d.ndim == 2:
            dTm = torch.matmul(d, m.T)
            dTmd = torch.sum(dTm * d, dim=1)
            return torch.sqrt(dTmd)
        else:
            raise ValueError(f"Unsupported shape {d.shape} for metric scale calculation.")
    
    def from_ECEF(self, xyz_ecef: torch.Tensor, *, is_point=False):
        if is_point:
            xyz_ecef = xyz_ecef - self.origin_ecef
        xyz_frame = torch.matmul(xyz_ecef, self.ecef_to_field_mat.T)
        return xyz_frame
    
    def to_ECEF(self, xyz_frame: torch.Tensor, *, is_point=False):
        xyz_ecef = torch.matmul(xyz_frame, self.field_to_ecef_mat.T)
        if is_point:
            xyz_ecef = xyz_ecef + self.origin_ecef
        return xyz_ecef
    
    def from_local_UNE(self, xyz_une: torch.Tensor, *, is_point=False):
        if is_point:
            xyz_une[..., 0] = xyz_une[..., 0] - self.origin_altitude
        xyz_frame = torch.matmul(xyz_une, self.une_to_field_mat.T)
        return xyz_frame

    def to_local_UNE(self, xyz_frame: torch.Tensor, *, is_point=False):
        xyz_une = torch.matmul(xyz_frame, self.field_to_une_mat.T)
        if is_point:
            xyz_une[..., 0] = xyz_une[..., 0] + self.origin_altitude
        return xyz_une

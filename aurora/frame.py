from typing import Union

import torch

import aurora.geodesy as geo


class Frame:
    def __init__(self,
        origin_latitude: float,
        origin_longitude: float,
        origin_altitude: float,
        field_inclination: float,
        field_declination: float
    ):
        # Get the frame transforms
        o_lat, o_lon, o_alt = origin_latitude, origin_longitude, origin_altitude
        o_ecef = geo.geodetic_to_ecef(o_lat, o_lon, o_alt)

        enu_to_ecef = torch.stack(geo.enu_basis_vectors_ecef(o_lat, o_lon)).T
        ecef_to_enu = torch.linalg.inv(enu_to_ecef)
        field_dir_enu = geo.inc_dec_to_enu(field_inclination, field_declination)
        z_dir_enu = -field_dir_enu
        x_dir_enu = torch.tensor([
            -field_dir_enu[0].item(),
            -field_dir_enu[1].item(),
             0.0
        ])
        x_dir_enu /= torch.linalg.vector_norm(x_dir_enu)
        y_dir_enu = torch.linalg.cross(z_dir_enu, x_dir_enu)
        field_to_enu = torch.tensor([
            x_dir_enu.tolist(),
            y_dir_enu.tolist(),
            z_dir_enu.tolist()
        ]).T
        enu_to_field = torch.linalg.inv(field_to_enu)
        ecef_to_field = torch.matmul(enu_to_field, ecef_to_enu)
        field_to_ecef = torch.matmul(enu_to_ecef, field_to_enu)

        # Store the reference frame parameters
        self.origin_ecef = o_ecef
        self.enu_to_field_mat = enu_to_field
        self.field_to_enu_mat = field_to_enu
        self.ecef_to_field_mat = ecef_to_field
        self.field_to_ecef_mat = field_to_ecef

        self.origin_latitude = o_lat
        self.origin_longitude = o_lon
        self.origin_altitude = o_alt

        self.field_inclination = field_inclination
        self.field_declination = field_declination

        self.z_to_h_factor = z_dir_enu[2].item()
    
    @property
    def device(self):
        return self.origin_ecef.device
    
    def altitude_to_z(self, h: Union[torch.Tensor, float]):
        return (h - self.origin_altitude) / self.z_to_h_factor
    
    def z_to_altitude(self, z: Union[torch.Tensor, float]):
        return z * self.z_to_h_factor + self.origin_altitude
    
    def from_ecef(self, xyz_ecef: torch.Tensor, *, is_point=False):
        if is_point:
            xyz_ecef = xyz_ecef - self.origin_ecef
        xyz_frame = torch.matmul(xyz_ecef, self.ecef_to_field_mat.T)
        return xyz_frame
    
    def to_ecef(self, xyz_frame: torch.Tensor, *, is_point=False):
        xyz_ecef = torch.matmul(xyz_frame, self.field_to_ecef_mat.T)
        if is_point:
            xyz_ecef = xyz_ecef + self.origin_ecef
        return xyz_ecef
    
    def from_local_enu(self, xyz_enu: torch.Tensor, *, is_point=False):
        if is_point:
            xyz_enu[..., 2] = xyz_enu[..., 2] - self.origin_altitude
        xyz_frame = torch.matmul(xyz_enu, self.enu_to_field_mat.T)
        return xyz_frame

    def to_local_enu(self, xyz_frame: torch.Tensor, *, is_point=False):
        xyz_enu = torch.matmul(xyz_frame, self.field_to_enu_mat.T)
        if is_point:
            xyz_enu[..., 2] = xyz_enu[..., 2] + self.origin_altitude
        return xyz_enu
    
    def __repr__(self):
        return (
            f"Frame("
            f"o_lat={self.origin_latitude}, "
            f"o_lon={self.origin_longitude}, "
            f"o_alt={self.origin_altitude}, "
            f"mf_inc={self.field_inclination}, "
            f"mf_dec={self.field_declination}"
            f")"
        )
    
    def to(self, *args, **kwargs):
        """Move all tensors to the specified device."""
        self.origin_ecef = self.origin_ecef.to(*args, **kwargs)
        self.enu_to_field_mat = self.enu_to_field_mat.to(*args, **kwargs)
        self.field_to_enu_mat = self.field_to_enu_mat.to(*args, **kwargs)
        self.ecef_to_field_mat = self.ecef_to_field_mat.to(*args, **kwargs)
        self.field_to_ecef_mat = self.field_to_ecef_mat.to(*args, **kwargs)
        return self

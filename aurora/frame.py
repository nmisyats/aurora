import torch

from aurora.geodesy import (
    lat_lon_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE,
    az_ze_to_UNE
)


class ReferenceFrame:
    def __init__(self,
        origin_latitude: float,
        origin_longitude: float,
        origin_altitude: float,
        field_inclination: float,
        field_declination: float,
        range_south: tuple[float, float],
        range_east: tuple[float, float],
        height: float,
        device: torch.device
    ):
        self.device = device
        
        # Get the frame transform
        o_lat, o_lon, o_alt = origin_latitude, origin_longitude, origin_altitude
        o_ecef_unit = lat_lon_to_ECEF(o_lat, o_lon)
        radius = earth_radius(o_lat, o_lon) + o_alt
        o_ecef = radius * o_ecef_unit
        o_ecef = o_ecef.to(device)

        une_to_ecef = torch.stack(UNE_basis_ECEF(o_lat, o_lon)).T
        ecef_to_une = torch.linalg.inv(une_to_ecef)
        field_dir_une = inc_dec_to_UNE(
            torch.scalar_tensor(field_inclination),
            torch.scalar_tensor(field_declination)
        )
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
        
        # Define the oblique reference frame bounding box
        x_min, x_max = range_south
        y_min, y_max = range_east
        h = height
        self.box_min = torch.tensor([x_min, y_min, 0.0], device=device)
        self.box_max = torch.tensor([x_max, y_max,   h], device=device)
        self.xy_min = self.box_min[:2]
        self.xy_max = self.box_max[:2]

        self.origin_altitude = torch.scalar_tensor(origin_altitude, device=device)
    
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
    
    def from_une(self, xyz_une: torch.Tensor, *, is_point=False):
        if is_point:
            xyz_une[..., 0] = xyz_une[..., 0] - self.origin_altitude
        xyz_frame = torch.matmul(xyz_une, self.une_to_field_mat.T)
        return xyz_frame

    def to_une(self, xyz_frame: torch.Tensor, *, is_point=False):
        xyz_une = torch.matmul(xyz_frame, self.field_to_une_mat.T)
        if is_point:
            xyz_une[..., 0] = xyz_une[..., 0] + self.origin_altitude
        return xyz_une
    
    def create_rays(self, o_lat: float, o_lon: float, o_alt: float, az: torch.Tensor, ze: torch.Tensor):
        az = az.flatten().to(self.device)
        ze = ze.flatten().to(self.device)
        rd_une = az_ze_to_UNE(az, ze).to(self.device)
        rd_rel = self.from_une(rd_une, is_point=False)

        ro_ecef_unit = lat_lon_to_ECEF(o_lat, o_lon).to(self.device)
        radius = earth_radius(o_lat, o_lon) + o_alt
        ro_ecef = radius * ro_ecef_unit
        ro_rel = self.from_ecef(ro_ecef, is_point=True)
        ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

        return ro_rel, rd_rel
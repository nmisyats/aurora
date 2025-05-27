import torch
import numpy as np
from aurora.utils import MinMax
from aurora.geodesy import (
    lat_lon_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE
)

class ReferenceFlux:
    def __init__(self, flux: torch.Tensor, oblique_range_x: MinMax, oblique_range_y: MinMax, device: torch.device):
        self.flux = flux.to(device)
        x_min, x_max = oblique_range_x
        y_min, y_max = oblique_range_y
        self.xy_min = torch.tensor([x_min, y_min], device=device)
        self.xy_max = torch.tensor([x_max, y_max], device=device)


class ReferenceFrame:
    def __init__(self,
        origin_latitude: float,
        origin_longitude: float,
        origin_altitude: float,
        field_inclination: float,
        field_declination: float,
        oblique_range_x: MinMax,
        oblique_range_y: MinMax,
        oblique_height: float,
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
        ecef_to_field = ecef_to_field.to(device)

        metric_tensor = torch.matmul(ecef_to_field, ecef_to_field.T)
        metric_tensor = metric_tensor.to(device)

        # Store the reference frame parameters
        self.origin_ecef = o_ecef
        self.ecef_to_field_mat = ecef_to_field
        self.metric_tensor = metric_tensor
        
        # Define the oblique reference frame bounding box
        x_min, x_max = oblique_range_x
        y_min, y_max = oblique_range_y
        h = oblique_height
        self.box_min = torch.tensor([x_min, y_min, 0.0], device=device)
        self.box_max = torch.tensor([x_max, y_max,   h], device=device)
        self.xy_min = self.box_min[:2]
        self.xy_max = self.box_max[:2]

        # TODO: scale by oblicity of reference frame z axis
        z_offset = origin_altitude
        self.z_offset = torch.scalar_tensor(z_offset, device=device)


class PhysicalModel:
    def __init__(
            self,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            emission_matrix: torch.Tensor,
            frame: ReferenceFrame,
            device: torch.device
        ):
        self.device = device
        self.frame = frame
        self.z_edges = altitude_bins.to(self.device)
        self.m_mat = emission_matrix.to(self.device)
        self.E_edges = energy_bins.to(self.device)

    def L(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        z = z + self.frame.z_offset
        z_idx = torch.bucketize(z.contiguous(), self.z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, self.m_mat.shape[1]-1)
        m_z = self.m_mat[z_idx,:]
        l = torch.sum(m_z * f, dim=1)
        return l
    
    def integrate_g(self,
          rd: torch.Tensor,
          t: torch.Tensor,
          p: torch.Tensor,
          f: torch.Tensor
        ) -> torch.Tensor:
        z = p[:, 2]
        l = self.L(z, f)
        g = torch.trapezoid(l, t)
        # dt = t[1:] - t[:-1]
        # l_lower, l_upper = l[:-1], l[1:]
        # l = (l_lower + l_upper) / 2.0
        # g = torch.sum(l * dt)
        g *= torch.sqrt(rd @ self.frame.metric_tensor @ rd)
        g /= 10.0
        return g
    
    def q0(self, f: torch.Tensor) -> torch.Tensor:
        e = 1.602e-19
        lower_E, upper_E = self.E_edges[:-1], self.E_edges[1:]
        E = (lower_E + upper_E) / 2.0
        dE = upper_E - lower_E
        q = (10**3) * e * (10**4) * np.pi * (f * E * dE)
        q = torch.sum(q, dim=1)
        return q

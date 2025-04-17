import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from matplotlib import pyplot as plt
import tqdm
from typing import Callable
from abc import ABC, abstractmethod

from aurora.data import Camera, Model
from aurora.geodesy import (
    lat_lon_to_ECEF,
    az_ze_to_UNE,
    UNE_to_ECEF,
    UNE_basis_ECEF,
    earth_radius,
    inc_dec_to_UNE
)
from aurora.geometry import ray_box_intersection


class Reconstruction(ABC):
    def __init__(self, pm: Model, f_net: nn.Module | None = None):
        o_ecef, ecef_to_field_mat = get_ray_transform(pm)
        self.origin_ecef = o_ecef
        self.ecef_to_field_mat = ecef_to_field_mat

        self.z_edges = torch.from_numpy(pm.altitude_bins).to(device)
        self.m_mat = torch.from_numpy(pm.emission_matrix).to(device)
        self.box_min = torch.tensor(pm.box_min).to(device)
        self.box_max = torch.tensor(pm.box_max).to(device)
        self.xy_min, self.xy_max = self.box_min[:2], self.box_max[:2]

        self.f_net = f_net

        self._vmap_g_cache = {}
    
    @property
    def device(self):
        return self.f_net.device

    @abstractmethod
    def f(self, xy: torch.Tensor) -> torch.Tensor:
        ...
    
    def L(self, p: torch.Tensor):
        xy, z = p[:,:2], p[:,2]
        z_idx = torch.bucketize(z.contiguous(), self.z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, self.m_mat.shape[1]-1)
        m_z = self.m_mat[z_idx,:]
        f = self.f(xy)
        l = torch.sum(m_z * f, dim=1)
        return l
    
    def g(self, ro: torch.Tensor, rd: torch.Tensor, tn: torch.Tensor, tf: torch.Tensor, bins: int):
        if bins not in self._vmap_g_cache:
            self._vmap_g_cache[bins] = self._make_vmap_g(bins)
        vmap_g = self._vmap_g_cache[bins]
        return vmap_g(ro, rd, tn, tf)

    def _make_vmap_g(self, bins: int):
        def g1(ro, rd, tn, tf):
            t, p = create_ray_points(ro, rd, tn, tf, bins)
            l = self.L(p)
            d = t[1:] - t[:-1]
            g = torch.sum(l[1:] * d) + l[0] * (tn - t[0])
            g /= 10.0
            return g
        return torch.vmap(g1, randomness='different')
    
    def train(self, cams: list[Camera], num_iters: int, batch_size: int, ray_bins: int):
        ro, rd, g_ref = create_ray_g_pairs(cams, self.origin_ecef, self.ecef_to_field_mat)
        tn, tf = ray_box_intersection(ro, rd, self.box_min, self.box_max)
        # Get mask for non-NaN values in tn
        valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))
        # Apply the mask to all tensors
        ro = ro[valid_mask].contiguous().to(self.device)
        rd = rd[valid_mask].contiguous().to(self.device)
        g_ref = g_ref[valid_mask].contiguous().to(self.device)
        tn = tn[valid_mask].contiguous().to(self.device)
        tf = tf[valid_mask].contiguous().to(self.device)

        optimizer = torch.optim.Adam(self.f_net.parameters(), lr=5e-5, weight_decay=1.0)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)

        losses = []

        tq = tqdm.trange(num_iters)
        for iter in tq:
            optimizer.zero_grad()

            idxs = torch.randint(0, len(ro), (batch_size,))
            g = self.g(ro[idxs], rd[idxs], tn[idxs], tf[idxs], ray_bins)
            loss = (g - g_ref[idxs])**2
            loss = loss.sum() / batch_size
            loss.backward()

            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            tq.set_postfix(
                loss=f"{loss.item():.0f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}"
            )
        tq.close()

        return losses


def get_ray_transform(pm: Model):
    o_lat, o_lon = pm.box_origin_lat, pm.box_origin_lon
    o_ecef = lat_lon_to_ECEF(o_lat, o_lon)

    une_to_ecef = torch.stack(UNE_basis_ECEF(o_lat, o_lon)).T
    ecef_to_une = torch.linalg.inv(une_to_ecef)
    field_dir_une = inc_dec_to_UNE(
        torch.scalar_tensor(pm.mag_field_inclination),
        torch.scalar_tensor(pm.mag_field_declination)
    )
    field_to_une = torch.stack((
        torch.tensor([0.0, -1.0, 0.0]),
        torch.tensor([0.0,  0.0, 1.0]),
        -field_dir_une
    )).T
    une_to_field = torch.linalg.inv(field_to_une)
    ecef_to_field = torch.matmul(une_to_field, ecef_to_une)

    return o_ecef, ecef_to_field

def create_camera_rays(cam: Camera, o_ecef: torch.Tensor, ecef_to_field: torch.Tensor):
    lat, lon = cam.latitude, cam.longitude

    az = torch.from_numpy(cam.azimuth).flatten()
    ze = torch.from_numpy(cam.zenith).flatten()
    rd_une = az_ze_to_UNE(az, ze)
    rd_ecef = UNE_to_ECEF(rd_une, lat, lon)
    rd_rel = torch.matmul(rd_ecef, ecef_to_field.T)

    radius = earth_radius(lat, lon) + cam.altitude
    ro_ecef = lat_lon_to_ECEF(lat, lon)
    ro_rel = radius * (ro_ecef - o_ecef)
    ro_rel = torch.matmul(ro_rel, ecef_to_field.T)
    ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

    return ro_rel, rd_rel

def create_ray_g_pairs(cams: list[Camera], o_ecef: torch.Tensor, ecef_to_field: torch.Tensor):
    ro_list, rd_list, g_ref_list = [], [], []
    for cam in cams:
        ro_rel, rd_rel = create_camera_rays(cam, o_ecef, ecef_to_field)
        ro_list.append(ro_rel)
        rd_list.append(rd_rel)
        
        g = torch.from_numpy(cam.image).flatten()
        g_ref_list.append(g)
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    g_ref = torch.cat(g_ref_list)
    
    return ro, rd, g_ref

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float, num_bins: int):
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1, device=device) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins, device=device) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd



import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Callable
from dataclasses import dataclass

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


@dataclass
class Frame:
    origin_ecef: torch.Tensor
    ecef_to_field_mat: torch.Tensor
    metric_tensor: torch.Tensor
    vol_min: torch.Tensor
    vol_max: torch.Tensor
    z_offset: torch.Tensor


class Reconstruction(ABC):
    def __init__(self, pm: Model, device: torch.device):
        self.device = device

        o_ecef, ecef_to_field, metric_tensor = get_ray_transform(pm, device)
        self.origin_ecef = o_ecef
        self.ecef_to_field_mat = ecef_to_field
        self.metric_tensor = metric_tensor

        self.z_edges = torch.from_numpy(pm.altitude_bins).to(self.device)
        self.m_mat = torch.from_numpy(pm.emission_matrix).to(self.device)
        
        vol = pm.volume
        vol_min = (vol.min.x, vol.min.y, vol.min.z)
        vol_max = (vol.max.x, vol.max.y, vol.max.z)
        self.vol_min = torch.tensor(vol_min).to(self.device)
        self.vol_max = torch.tensor(vol_max).to(self.device)
        self.xy_min = self.vol_min[:2]
        self.xy_max = self.vol_max[:2]
        self.z_offset = torch.scalar_tensor(vol.alt).to(device)

        self._vmap_g_cache = {}

    @property
    def frame(self):
        return Frame(
            origin_ecef=self.origin_ecef,
            ecef_to_field_mat=self.ecef_to_field_mat,
            metric_tensor=self.metric_tensor,
            vol_min=self.vol_min,
            vol_max=self.vol_max,
            z_offset=self.z_offset
        )

    @abstractmethod
    def parameters() -> Iterable[nn.Parameter]:
        ...

    @abstractmethod
    def f(self, xy: torch.Tensor) -> torch.Tensor:
        ...
    
    def L(self, p: torch.Tensor) -> torch.Tensor:
        xy, z = p[:,:2], p[:,2]
        z = z + self.z_offset
        z_idx = torch.bucketize(z.contiguous(), self.z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, self.m_mat.shape[1]-1)
        m_z = self.m_mat[z_idx,:]
        f = self.f(xy)
        l = torch.sum(m_z * f, dim=1)
        return l
    
    def train(self, cams: list[Camera], num_iters: int, batch_size: int, ray_bins: int):
        # Create rays and compute intersections
        ro, rd, g_ref = create_ray_g_pairs(cams, self.origin_ecef, self.ecef_to_field_mat, self.device)
        tn, tf = ray_box_intersection(ro, rd, self.vol_min, self.vol_max)
        # Get mask for non-NaN values in tn
        valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))
        # Apply the mask to all tensors
        ro = ro[valid_mask].contiguous()
        rd = rd[valid_mask].contiguous()
        g_ref = g_ref[valid_mask].contiguous()
        tn = tn[valid_mask].contiguous()
        tf = tf[valid_mask].contiguous()

        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5, weight_decay=1.0) # change to -4
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1) # step size 500

        renderer = Renderer(self.L, self.frame)

        losses = []

        tq = tqdm.trange(num_iters)
        for iter in tq:
            optimizer.zero_grad()

            idxs = torch.randint(0, len(ro), (batch_size,))
            g = renderer.g(ro[idxs], rd[idxs], tn[idxs], tf[idxs], ray_bins)
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


class Renderer:
    def __init__(self, L: Callable, frame: Frame):
        self.L = L
        self.frame = frame
        self.device = frame.origin_ecef.device
        
        self._vmap_g_cache = {}

    def g(self, ro: torch.Tensor, rd: torch.Tensor, tn: torch.Tensor, tf: torch.Tensor, ray_bins: int):
        if ray_bins not in self._vmap_g_cache:
            self._vmap_g_cache[ray_bins] = self._make_vmapped_g(ray_bins)
        vmap_g = self._vmap_g_cache[ray_bins]
        return vmap_g(ro, rd, tn, tf)

    def _make_vmapped_g(self, ray_bins: int):
        def single_ray_g(ro, rd, tn, tf):
            t, p = create_ray_points(ro, rd, tn, tf, ray_bins, self.device)
            l = self.L(p)
            d = t[1:] - t[:-1]
            g = torch.sum(l[1:] * d) + l[0] * (tn - t[0])
            g *= torch.sqrt(rd @ self.frame.metric_tensor @ rd)
            g /= 10.0
            return g
        return torch.vmap(single_ray_g, randomness='different')
    
    def image(self, cam: Camera, ray_bins: int, nan=0.0):
        ro, rd = create_camera_rays(cam, self.frame.origin_ecef, self.frame.ecef_to_field_mat, self.device)
        tn, tf = ray_box_intersection(ro, rd, self.frame.vol_min, self.frame.vol_max)
        g = self.g(ro, rd, tn, tf, ray_bins)
        h, w = cam.image.shape
        img = g.reshape(h, w)
        img = torch.nan_to_num(img, nan=nan)
        return img


def get_ray_transform(pm: Model, device: torch.device):
    vol = pm.volume
    o_lat, o_lon, o_alt = vol.lat, vol.lon, vol.alt
    o_ecef_unit = lat_lon_to_ECEF(o_lat, o_lon)
    radius = earth_radius(o_lat, o_lon) + o_alt
    o_ecef = radius * o_ecef_unit
    o_ecef = o_ecef.to(device)

    une_to_ecef = torch.stack(UNE_basis_ECEF(o_lat, o_lon)).T
    ecef_to_une = torch.linalg.inv(une_to_ecef)
    field_dir_une = inc_dec_to_UNE(
        torch.scalar_tensor(pm.field.inc),
        torch.scalar_tensor(pm.field.dec)
    )
    field_to_une = torch.stack((
        torch.tensor([0.0, -1.0, 0.0]),
        torch.tensor([0.0,  0.0, 1.0]),
        -field_dir_une
    )).T
    une_to_field = torch.linalg.inv(field_to_une)
    ecef_to_field = torch.matmul(une_to_field, ecef_to_une)
    ecef_to_field = ecef_to_field.to(device)

    # metric tensor
    metric_tensor = torch.matmul(ecef_to_field, ecef_to_field.T)
    metric_tensor = metric_tensor.to(device)

    return o_ecef, ecef_to_field, metric_tensor

def create_camera_rays(cam: Camera, o_ecef: torch.Tensor, ecef_to_field: torch.Tensor, device: torch.device):
    lat, lon, alt = cam.latitude, cam.longitude, cam.altitude

    az = torch.from_numpy(cam.azimuth).flatten().to(device)
    ze = torch.from_numpy(cam.zenith).flatten().to(device)
    rd_une = az_ze_to_UNE(az, ze).to(device)
    rd_ecef = UNE_to_ECEF(rd_une, lat, lon).to(device)
    rd_rel = torch.matmul(rd_ecef, ecef_to_field.T)

    ro_ecef_unit = lat_lon_to_ECEF(lat, lon).to(device)
    radius = earth_radius(lat, lon) + alt
    ro_ecef = radius * ro_ecef_unit
    ro_ecef_rel = ro_ecef - o_ecef
    ro_rel = torch.matmul(ro_ecef_rel, ecef_to_field.T)
    ro_rel = ro_rel.repeat(rd_rel.shape[0], 1)

    return ro_rel, rd_rel

def create_ray_g_pairs(cams: list[Camera], o_ecef: torch.Tensor, ecef_to_field: torch.Tensor, device: torch.device):
    ro_list, rd_list, g_ref_list = [], [], []
    for cam in cams:
        ro_rel, rd_rel = create_camera_rays(cam, o_ecef, ecef_to_field, device)
        ro_list.append(ro_rel)
        rd_list.append(rd_rel)
        
        g = torch.from_numpy(cam.image).flatten().to(device)
        g_ref_list.append(g)
    
    ro = torch.cat(ro_list)
    rd = torch.cat(rd_list)
    g_ref = torch.cat(g_ref_list)
    
    return ro, rd, g_ref

def create_ray_points(ro: torch.Tensor, rd: torch.Tensor, min_t: float, max_t: float, num_bins: int, device: torch.device):
    bin_edges = torch.linspace(min_t, max_t, num_bins + 1, device=device) # num_bins + 1 edges
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:-1]
    upper_edges = bin_edges[1:]
    # Generate random values in each bin
    t = lower_edges + torch.rand(num_bins, device=device) * (upper_edges - lower_edges)
    # Create points
    return t, ro + t.reshape(t.shape[0], 1) * rd
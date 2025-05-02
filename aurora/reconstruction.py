import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from abc import ABC, abstractmethod
from typing import Iterable, Callable
from dataclasses import dataclass

from aurora.data import Camera, Model
from aurora.geometry import (
    get_frame_transform,
    create_camera_rays,
    create_ray_g_pairs,
    create_ray_points,
    ray_box_intersection
)


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

        o_ecef, ecef_to_field, metric_tensor = get_frame_transform(pm, device)
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
        self.z_offset = torch.scalar_tensor(vol.alt).to(device) # TODO: scale by oblicity of reference frame z axis

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

        renderer = Renderer(self.L, self.frame, self.device)

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
    def __init__(self, L: Callable, frame: Frame, device: torch.device):
        self.L = L
        self.frame = frame
        self.device = device
        
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

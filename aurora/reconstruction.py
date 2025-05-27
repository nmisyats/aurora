import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import tqdm
from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path

from aurora.camera import Camera
from aurora.data import (
    PhysicalModelData,
    ReferenceFrameDescription,
    VolumeDescription
)
from aurora.geometry import (
    get_frame_transform,
    create_camera_rays,
    create_ray_points,
    ray_box_intersection
)


class ReferenceFrame:
    def __init__(self, frame_desc: ReferenceFrameDescription, volume_desc: VolumeDescription, device: torch.device):
        self.device = device

        # Get the frame transform
        o_ecef, ecef_to_field, metric_tensor = get_frame_transform(
            frame_desc.origin_latitude,
            frame_desc.origin_longitude,
            frame_desc.origin_altitude,
            frame_desc.field_inclination,
            frame_desc.field_declination,
            device
        )
        self.origin_ecef = o_ecef
        self.ecef_to_field_mat = ecef_to_field
        self.metric_tensor = metric_tensor
        
        x_min, x_max = volume_desc.oblique_range_x
        y_min, y_max = volume_desc.oblique_range_y
        h = volume_desc.oblique_height
        self.box_min = torch.tensor([x_min, y_min, 0.0], device=device)
        self.box_max = torch.tensor([x_max, y_max,   h], device=device)
        self.xy_min = self.box_min[:2]
        self.xy_max = self.box_max[:2]

        # TODO: scale by oblicity of reference frame z axis
        z_offset = frame_desc.origin_altitude
        self.z_offset = torch.scalar_tensor(z_offset, device=device)

class PhysicalModel:
    def __init__(self, pm_data: PhysicalModelData, device: torch.device):
        self.device = device
        self.frame = ReferenceFrame(
            pm_data.reference_frame,
            pm_data.bounding_volume,
            device
        )
        self.z_edges = torch.from_numpy(pm_data.altitude_bins).to(self.device)
        self.m_mat = torch.from_numpy(pm_data.emission_matrix).to(self.device)
        self.E_edges = torch.from_numpy(pm_data.energy_bins).to(self.device)
    
    def q0(self, f: torch.Tensor) -> torch.Tensor:
        e = 1.602e-19
        lower_E, upper_E = self.E_edges[:-1], self.E_edges[1:]
        E = (lower_E + upper_E) / 2.0
        dE = upper_E - lower_E
        q = (10**3) * e * (10**4) * np.pi * (f * E * dE)
        q = torch.sum(q, dim=1)
        return q

class RayDataset:
    def __init__(self, cams: list[Camera], pm: PhysicalModel, device: torch.device):
        self.physical_model = pm
        self.frame = pm.frame
        self.device = device

        # Create rays and compute intersections
        ro, rd, g_ref = self._create_ray_g_pairs(cams)
        tn, tf = ray_box_intersection(ro, rd, self.frame.box_min, self.frame.box_max)
        # Get mask for non-NaN values in tn
        valid_mask = ~(torch.isnan(tn) | torch.isnan(tf))
        # Apply the mask to all tensors
        self.ro = ro[valid_mask].contiguous()
        self.rd = rd[valid_mask].contiguous()
        self.g_ref = g_ref[valid_mask].contiguous()
        self.tn = tn[valid_mask].contiguous()
        self.tf = tf[valid_mask].contiguous()

    def _create_ray_g_pairs(self, cams: list[Camera]):
        ro_list, rd_list, g_ref_list = [], [], []
        for cam in cams:
            ro_rel, rd_rel = create_camera_rays(
                cam,
                self.frame.origin_ecef,
                self.frame.ecef_to_field_mat,
                self.device)
            ro_list.append(ro_rel)
            rd_list.append(rd_rel)
            
            g = torch.from_numpy(cam.image).flatten().to(self.device)
            g_ref_list.append(g)
        
        ro = torch.cat(ro_list)
        rd = torch.cat(rd_list)
        g_ref = torch.cat(g_ref_list)
        
        return ro, rd, g_ref

    def __len__(self):
        return len(self.ro)
    
    def __getitem__(self, idx):
        return (
            self.ro[idx], self.rd[idx], self.g_ref[idx], self.tn[idx], self.tf[idx]
        )

class Reconstruction(ABC):
    def __init__(self, pm: PhysicalModel, device: torch.device):
        self.physical_model = pm
        self.frame = pm.frame
        self.device = device

        self._vmap_g_cache = {}
    
    @abstractmethod
    def eval_mode(self):
        ...
    
    @abstractmethod
    def train_mode(self):
        ...

    @abstractmethod
    def parameters() -> Iterable[nn.Parameter]:
        ...

    @abstractmethod
    def f(self, xy: torch.Tensor) -> torch.Tensor:
        ...

    def L(self, p: torch.Tensor) -> torch.Tensor:
        xy, z = p[:,:2], p[:,2]
        z = z + self.frame.z_offset
        z_edges = self.physical_model.z_edges
        m_mat = self.physical_model.m_mat
        z_idx = torch.bucketize(z.contiguous(), z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, m_mat.shape[1]-1)
        m_z = m_mat[z_idx,:]
        f = self.f(xy)
        l = torch.sum(m_z * f, dim=1)
        return l
    
    def g(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            tn: torch.Tensor, 
            tf: torch.Tensor, 
            ray_bins: int
        ) -> torch.Tensor:
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
            met_t = self.frame.metric_tensor
            g *= torch.sqrt(rd @ met_t @ rd)
            g /= 10.0
            return g
        return torch.vmap(single_ray_g, randomness='different')
    
    def image(self, cam: Camera, ray_bins: int, nan=0.0) -> torch.Tensor:
        ro, rd = create_camera_rays(
            cam,
            self.frame.origin_ecef,
            self.frame.ecef_to_field_mat,
            self.device)
        tn, tf = ray_box_intersection(ro, rd, self.frame.box_min, self.frame.box_max)
        g = self.g(ro, rd, tn, tf, ray_bins)
        h, w = cam.image.shape
        img = g.reshape(h, w)
        img = torch.nan_to_num(img, nan=nan)
        return img

    def image_renderer(self, ray_bins: int, nan=0.0):
        return lambda cam: self.image(cam, ray_bins, nan)
    
    def train(self,
            dataset: RayDataset,
            num_iters: int,
            batch_size: int,
            ray_bins: int = 100,
            lr: float = 5e-5, # change to -4
            weight_decay: float = 1.0,
            lr_step_size: int = 5000, # step size 500
            lr_gamma: float = 0.1
        ) -> list[float]:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        losses = []

        tq = tqdm.trange(num_iters)
        for iter in tq:
            optimizer.zero_grad()

            idxs = torch.randint(0, len(dataset), (batch_size,))
            ro, rd, g_ref, tn, tf = dataset[idxs]
            g = self.g(ro, rd, tn, tf, ray_bins)
            loss = (g - g_ref)**2
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
    
    @abstractmethod
    def to_dict(self) -> dict:
        ...
    
    @classmethod
    @abstractmethod
    def from_dict(cls, recon_dict: dict, device: torch.device) -> 'Reconstruction':
        ...
    
    def save(self, path: Path | str):
        torch.save(self.to_dict(), path)
    
    @classmethod
    def load(cls, path: Path | str, device: torch.device):
        recon_dict = torch.load(path, map_location=device, weights_only=False)
        return cls.from_dict(recon_dict, device)
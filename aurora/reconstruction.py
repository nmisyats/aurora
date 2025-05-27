import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path

from aurora.camera import Camera
from aurora.geometry import (
    create_camera_rays,
    create_ray_points,
    ray_box_intersection
)
from aurora.physics import ReferenceFrame, PhysicalModel


class RayDataset:
    def __init__(self, cams: list[Camera], frame: ReferenceFrame, device: torch.device):
        self.frame = frame
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
            
            g = cam.image.flatten().to(self.device)
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
        f = self.f(xy)
        return self.physical_model.L(z, f)
    
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
            xy = p[:,:2]
            f = self.f(xy)
            return self.physical_model.integrate_g(rd, t, p, f)
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
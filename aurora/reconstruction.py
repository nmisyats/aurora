import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from abc import ABC, abstractmethod
from typing import Iterable
from pathlib import Path

from aurora.camera import Camera
from aurora.geometry import (
    create_ray_points,
    ray_box_intersection
)
from aurora.physics import ReferenceFrame, PhysicalModel
from aurora.utils import MinMax


class ElectronFluxModel(ABC):
    @abstractmethod
    def f_at(self, xy: torch.Tensor) -> torch.Tensor:
        ...


class TrainableFluxModel(ElectronFluxModel, nn.Module):
    def f_at(self, xy: torch.Tensor):
        return self.forward(xy)


class ReferenceFlux(ElectronFluxModel):
    def __init__(self, image: torch.Tensor, oblique_range_x: MinMax, oblique_range_y: MinMax, device: torch.device):
        self.image = image.to(device)
        self.device = device
        x_min, x_max = oblique_range_x
        y_min, y_max = oblique_range_y
        self.xy_min = torch.tensor([x_min, y_min], device=device)
        self.xy_max = torch.tensor([x_max, y_max], device=device)
    
    def f_at(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (N, 2) coordinates within xy_min and xy_max
        # self.image: (H, W, B)
        # Output: (N, B) sampled flux at each xy

        H, W, B = self.image.shape

        # Normalize xy to [0, 1]
        norm_xy = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        norm_xy = torch.clamp(norm_xy, 0, 1)

        # Scale to image pixel coordinates
        y_idx = norm_xy[:, 1] * (H - 1)
        x_idx = norm_xy[:, 0] * (W - 1)

        # Create grid for grid_sample
        grid = torch.stack((x_idx, y_idx), dim=1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        grid = 2 * grid / torch.tensor([W - 1, H - 1], device=self.device) - 1
        grid = grid[..., [1, 0]]  # switch x, y -> y, x
        grid = grid.expand(B, -1, -1, -1)  # Expand to match flux shape

        # Prepare input image tensor for grid_sample
        flux = self.image.permute(2, 0, 1).unsqueeze(1)  # (B, 1, H, W)

        # Perform bilinear sampling
        sampled = torch.nn.functional.grid_sample(
            flux, grid, mode='bilinear', align_corners=True
        )  # (B, 1, N, 1)

        # Reshape result to (N, B)
        return sampled.squeeze(3).squeeze(1).T  # (N, B)


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
            cam_ro, cam_rd = cam.create_rays(self.frame, self.device)
            ro_list.append(cam_ro)
            rd_list.append(cam_rd)

            cam_g_ref = cam.image.flatten().to(self.device)
            g_ref_list.append(cam_g_ref)
        
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
    
    def sample(self, num_samples: int):
        if num_samples > len(self):
            raise ValueError("Number of samples exceeds dataset size.")
        idxs = torch.randint(0, len(self), (num_samples,))
        return self[idxs]


class Reconstruction:
    def __init__(self, pm: PhysicalModel, f_model: ElectronFluxModel, device: torch.device):
        self.physical_model = pm
        self.f_model = f_model
        self.frame = pm.frame
        self.device = device

        self._vmap_g_cache = {}

    def f(self, xy: torch.Tensor) -> torch.Tensor:
        return self.f_model.f_at(xy)

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
    
    def image(self, cam: Camera, ray_bins: int, nan=0.0):
        ro, rd = cam.create_rays(self.frame, self.device)
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

        if not isinstance(self.f_model, TrainableFluxModel):
            raise ValueError("Flux model is not trainable.")

        optimizer = torch.optim.Adam(self.f_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

        losses = []

        tq = tqdm.trange(num_iters)
        for iter in tq:
            optimizer.zero_grad()

            ro, rd, g_ref, tn, tf = dataset.sample(batch_size)
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
    def eval_mode(self):
        if isinstance(self.f_model, TrainableFluxModel):
            self.f_model.eval()
    
    @abstractmethod
    def train_mode(self):
        if isinstance(self.f_model, TrainableFluxModel):
            self.f_model.train()
    
    def save(self, path: Path | str):
        torch.save({
            "physical_model": self.physical_model,
            "f_model": self.f_model
        }, path)
    
    @classmethod
    def load(cls, path: Path | str, device: torch.device):
        data = torch.load(path, map_location=device, weights_only=False)
        return cls(data["physical_model"], data["f_model"], device)
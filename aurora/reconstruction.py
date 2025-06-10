from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torch.optim.lr_scheduler as lr_scheduler
import tqdm

from aurora.camera import Camera
from aurora.models import FluxModel, TrainableFluxModel
from aurora.frame import Frame
from aurora.bbox import BBox
from aurora.dataset import CameraRaysDataset, RadarPointsDataset
import aurora.geometry as geom
import aurora.physics as phy

class Reconstruction:
    def __init__(
        self,
        flux_model: FluxModel,
        frame: Frame,
        bbox: BBox,
        M_emis: torch.Tensor,
        M_dens: torch.Tensor,
        z_edges: torch.Tensor
    ):
        self.flux_model = flux_model
        self.M_emis = M_emis
        self.M_dens = M_dens
        self.z_edges = z_edges

        self.frame = frame
        self.bbox = bbox
        self.device = frame.device
        
        self._vmap_g_cache = {}
        self._training = True

    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron flux at points xy.
        Args:
            xy (torch.Tensor): Tensor of shape (..., 2) in South-East coordinates.
        
        Returns:
            torch.Tensor: Flux tensor at the points xy of shape (..., n_E)
                where n_E is the number of energy bins.
        """
        if not self._training:
            with torch.no_grad():
                return self.flux_model.flux(xy)
        else:
            return self.flux_model.flux(xy)

    def emis_rate(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the emission rate at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (..., 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (...,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.emis_rate(z, f, self.M_emis, self.z_edges)
    
    def elec_dens(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron density at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (..., 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (...,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.elec_dens(z, f, self.M_dens, self.z_edges)

    def int_emis_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            tn: torch.Tensor, 
            tf: torch.Tensor,
            ray_bins: int
        ) -> torch.Tensor:
        """
        Integrate the emission along rays along a ray.
        
        Args:
            ray_bins (int): Number of bins to use for ray integration.
            ro (torch.Tensor): Ray origins of shape (n, ..., 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, ..., 3) in frame coordinates.
            tn (torch.Tensor): Near intersection distances of shape (n, ...,).
            tf (torch.Tensor): Far intersection distances of shape (n, ...,).
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n, ...,).
        """
        if ro.ndim == 1:
            ro = ro.unsqueeze(0)
            rd = rd.unsqueeze(0)
            tn = tn.unsqueeze(0)
            tf = tf.unsqueeze(0)
        if ray_bins not in self._vmap_g_cache:
            self._vmap_g_cache[ray_bins] = self._make_vmapped_g(ray_bins)
        vmap_g = self._vmap_g_cache[ray_bins]
        return vmap_g(ro, rd, tn, tf)

    def _make_vmapped_g(self, ray_bins: int):
        def int_single_ray(ro, rd, tn, tf):
            t_frame, p_frame = geom.create_ray_points(ro, rd, tn, tf, ray_bins, self.device)
            p_enu = self.frame.to_local_enu(p_frame, is_point=True)
            xy, z = p_frame[:,:2], p_enu[:,2]
            f = self.flux(xy)
            l = phy.emis_rate(z, f, self.M_emis, self.z_edges)
            g = phy.int_emis_rayleigh(t_frame, l)
            g = g # * self.frame.metric_scale(rd) not required?
            return g
        return torch.vmap(int_single_ray, randomness='different')
    
    @torch.no_grad()
    def image(self, cam: Camera, ray_bins: int, nan=0.0):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            ray_bins (int): Number of bins to use for ray integration.
            nan (float): Value to replace NaN values in the image.
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.frame.from_ecef(ro, is_point=True)
        rd = self.frame.from_ecef(rd, is_point=False)
        tn, tf = geom.ray_box_intersection(ro, rd, self.bbox.xyz_min, self.bbox.xyz_max)
        g = self.int_emis_ray(ro, rd, tn, tf, ray_bins)
        h, w = cam.image.shape
        img = g.reshape(h, w)
        img = torch.nan_to_num(img, nan=nan)
        return img
    
    def train(self, *,
            ray_data: CameraRaysDataset | None = None,
            radar_data: RadarPointsDataset | None = None,
            num_iters: int = 1000,
            ray_batch_size: int = 4096,
            ray_bins: int = 100,
            radar_batch_size: int = 1000,
            ray_loss_weight: float = 1.0,
            radar_loss_weight: float = 1.0,
            lr: float = 5e-5,
            weight_decay: float = 1.0,
            lr_step_size: int = 5000,
            lr_gamma: float = 0.1
        ) -> list[float]:
        
        if ray_data is None and radar_data is None:
            raise ValueError("At least one of ray_data or radar_data must be provided.")
        
        if not self._training:
            raise ValueError("Reconstruction not in training mode.")

        if not isinstance(self.flux_model, TrainableFluxModel):
            raise ValueError("Flux model is not trainable.")

        optimizer = torch.optim.Adam(self.flux_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        losses = []

        tq = tqdm.trange(num_iters)
        for iter in tq:
            optimizer.zero_grad()

            ray_loss, radar_loss = 0.0, 0.0

            if ray_data is not None and ray_loss_weight > 0.0:
                ro, rd, tn, tf, g_ref = ray_data.random_sample(ray_batch_size)
                g = self.int_emis_ray(ro, rd, tn, tf, ray_bins)
                ray_loss = (g - g_ref)**2
                ray_loss = ray_loss.sum() / ray_batch_size
            
            if radar_data is not None and radar_loss_weight > 0.0:
                p, d_ref = radar_data.random_sample(radar_batch_size)
                d = self.elec_dens(p)
                radar_loss = (d - d_ref)**2
                radar_loss = radar_loss.sum() / radar_batch_size

            loss = ray_loss_weight*ray_loss + radar_loss_weight*radar_loss
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
    
    def eval_mode(self):
        if isinstance(self.flux_model, TrainableFluxModel):
            self.flux_model.eval()
        self._training = False
    
    def train_mode(self):
        if isinstance(self.flux_model, TrainableFluxModel):
            self.flux_model.train()
        self._training = True


# def save_reonstruction(recon: Reconstruction, path: Path | str):
#     torch.save({
#         "flux_model": recon.flux_model,
#         "bbox": recon.bbox,
#         "M_emis": recon.M_emis,
#         "M_dens": recon.M_dens,
#         "z_edges": recon.z_edges
#     }, path)

# def load_reconstruction(path: Path | str, device: torch.device):
#     data = torch.load(path, map_location=device, weights_only=False)
#     return Reconstruction(
#         data["flux_model"],
#         data["bbox"],
#         data["M_emis"],
#         data["M_dens"],
#         data["z_edges"]
#     )
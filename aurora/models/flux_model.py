from pathlib import Path
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import tqdm

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.geometry as geom
import aurora.physics as phy
import aurora.data as data
from aurora.utils import normalize_batch_dims, ceiled_div, iter_chunks
from aurora.sampling import sample_uniform

class FluxModel(nn.Module, ABC):
    def __init__(
            self,
            frame: Frame,
            bbox: BBox,
            emis_mat: torch.Tensor,
            dens_mat: torch.Tensor,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
        ):
        super().__init__()

        self.frame = frame
        self.bbox = bbox
        
        self.register_buffer("emis_mat", emis_mat)
        self.register_buffer("dens_mat", dens_mat)
        self.register_buffer("altitude_bins", altitude_bins)
        self.register_buffer("energy_bins", energy_bins)

        self.chunk_size = None
        self.chunk_progress_bar = False
    
    @property
    def device(self):
        return self.emis_mat.device
    
    def _normalize_xy(self, xy: torch.Tensor):
        xy_min = self.bbox.xy_min
        xy_max = self.bbox.xy_max
        return (xy - xy_min) / (xy_max - xy_min)
    
    def _normalize_energy(self, E: torch.Tensor):
        E_min = self.energy_bins[0]
        E_max = self.energy_bins[-1]
        return (E - E_min) / (E_max - E_min)
    
    @abstractmethod
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron flux distribution at points xy.
        Args:
            xy (torch.Tensor): Tensor of shape (n, 2) in South-East coordinates.
        
        Returns:
            torch.Tensor: Flux tensor at the points xy of shape (n, n_E)
                where n_E is the number of energy bins.
        """
        ...

    @normalize_batch_dims({"xy": 1}, 0)
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron flux distribution at points xy. See `forward`.
        """
        if self.training:
            return self._forward_flux_train(xy)
        else:
            return self._forward_flux_eval(xy)
    
    def _forward_flux_train(self, xy: torch.Tensor) -> torch.Tensor:
        return self.forward(xy)
    
    @torch.no_grad()
    def _forward_flux_eval(self, xy: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None:
            return self._eval_flux_chunked(xy)
        else:
            return self.forward(xy)

    def _eval_flux_chunked(self, xy: torch.Tensor) -> torch.Tensor:
        f_chunks = []
        num_chunks = ceiled_div(len(xy), self.chunk_size)

        progress_bar = self.chunk_progress_bar
        iterator = tqdm.trange(num_chunks) if progress_bar else range(num_chunks)

        chunks = iter_chunks(len(xy), self.chunk_size)
        
        for _, (chunk_start, chunk_stop) in zip(iterator, chunks):
            xy_chunk = xy[chunk_start:chunk_stop]
            f_chunk = self.forward(xy_chunk)
            f_chunks.append(f_chunk)
            
            if progress_bar:
                iterator.set_postfix_str(f"flux_evals:{chunk_stop}/{len(xy)}")
        
        return torch.cat(f_chunks, dim=0)

    @normalize_batch_dims({"p_frame": 1}, 0)
    def emis_rate(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the emission rate at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (n,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.emis_rate(z, f, self.emis_mat, self.altitude_bins)
    
    @normalize_batch_dims({"p_frame": 1}, 0)
    def elec_dens(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron density at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in frame coordinates.
        
        Returns:
            torch.Tensor: Emission rate tensor at the points p of shape (n,).
        """
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        return phy.elec_dens(z, f, self.dens_mat, self.altitude_bins)

    @normalize_batch_dims({"ro": 1, "rd": 1, "tn": 0, "tf": 0}, 0)
    def int_emis_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            t: torch.Tensor
        ) -> torch.Tensor:
        """
        Integrate the emission along rays along a ray.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in frame coordinates.
            t (torch.Tensor): (n, n_sample) or (n_samples,) distances to sample points along the rays.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
        p_frame = geom.sample_ray_points(ro, rd, t)
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        l = phy.emis_rate(z, f, self.emis_mat, self.altitude_bins)
        g = phy.int_emis_rayleigh(t, l)
        return g
    
    @torch.no_grad()
    def image(self, cam: Camera, num_samples: int, nan=0.0):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            num_bins (int): Number of bins to use for ray integration.
            nan (float): Value to replace NaN values in the image.
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.frame.from_ecef(ro, is_point=True) # (n, 3)
        rd = self.frame.from_ecef(rd, is_point=False) # (n, 3)
        
        tn, tf = geom.ray_box_intersection(ro, rd, self.bbox.xyz_min, self.bbox.xyz_max)
        t = sample_uniform(tn, tf, num_samples)
        
        g = self.int_emis_ray(ro, rd, t)
        h, w = cam.image.shape
        img = g.reshape(h, w)
        img = torch.nan_to_num(img, nan=nan)
        return img
    
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        
        device = self._get_device_from_args(*args, **kwargs)
        if device is not None:
            result.frame = self.frame.to(device)
            result.bbox = self.bbox.to(device)
        
        return result
    
    def _get_device_from_args(self, *args, **kwargs):
        """Extract device from .to() arguments."""
        if args:
            arg = args[0]
            if isinstance(arg, (torch.device, str)):
                return arg
            elif hasattr(arg, 'device'): # tensor-like
                return arg.device
        if 'device' in kwargs:
            return kwargs['device']
        return None


# def save_reconstruction(recon: Reconstruction, path: Path | str):
#     torch.save({
#         "flux_model": recon.flux_model,
#         "frame": recon.frame,
#         "bbox": recon.bbox,
#         "emis_mat": recon.emis_mat,
#         "dens_mat": recon.dens_mat,
#         "altitude_bins": recon.altitude_bins
#     }, path)

# def load_reconstruction(path: Path | str, device: torch.device):
#     data = torch.load(path, map_location=device, weights_only=False)
#     return Reconstruction(
#         data["flux_model"],
#         data["frame"],
#         data["bbox"],
#         data["emis_mat"],
#         data["dens_mat"],
#         data["altitude_bins"]
#     )

# def load_static_reconstruction(flux_path: Path | str, config_path: Path | str, device: torch.device):
#     flux_path = Path(flux_path)
#     config_path = Path(config_path)
#     config = data.load_config(config_path, device)
#     return Reconstruction(
#         flux_model=GridSampledFlux(
#             xy_min=config.bbox.xy_min,
#             xy_max=config.bbox.xy_max,
#             energy_bins=config.physics.energy_bins,
#             data=data.load_3d_grid_data(flux_path).to(device)
#         ).to(device),
#         frame=config.frame,
#         bbox=config.bbox,
#         emis_mat=config.physics.emis_mat,
#         dens_mat=config.physics.dens_mat,
#         altitude_bins=config.physics.altitude_bins
#     ).eval()
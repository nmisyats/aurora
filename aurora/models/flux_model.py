from abc import ABC, abstractmethod
from typing import overload, Optional, Union, Dict
from dataclasses import dataclass

import torch
import torch.nn as nn
import tqdm

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
from aurora.physics import PhysicalModel
import aurora.geometry as gmt
import aurora.physics as phy
from aurora.utils import (
    normalize_batch_dims,
    ceiled_div,
    iter_chunks,
    flatten_batch_dims,
    unflatten_batch_dims
)
from aurora.samplers import RaySampler, EqualSampler


@dataclass
class ModelConfig:
    frame: Frame
    bbox: BBox
    physics: PhysicalModel


class FluxModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.frame = config.frame
        self.bbox = config.bbox
        
        self.register_buffer("emis_mats", config.physics.emis_mats)
        self.register_buffer("dens_mat", config.physics.dens_mat)
        self.register_buffer("altitude_bins", config.physics.altitude_bins)
        self.register_buffer("energy_bins", config.physics.energy_bins)
        self.wl_to_idx = config.physics.wl_to_idx

        # Compute the altitude bin edges in oblique frame
        z_bins = self.frame.altitude_to_z(config.physics.altitude_bins)
        self.register_buffer("z_bins", z_bins)

        self.num_bins = len(config.physics.energy_bins) - 1

        self.chunk_size = None
        self.chunk_progress_bar = False
    
    @property
    def device(self):
        return self.emis_mats.device
    
    @abstractmethod
    def forward(self, xy: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate the electron flux distribution at points xy at return
        relevant output.
        
        Args:
            xy (torch.Tensor): Tensor of shape (n, 2) in South-East coordinates.
        
        Returns:
            out: Dictionary of tensors. out["f"] is the flux tensor at the
            points xy of shape (n, n_E) where n_E is the number of
            energy bins.
        """
        ...

    @normalize_batch_dims(xy=1)
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron flux distribution at points xy. See `forward`.
        """
        if self.training:
            return self._forward_flux_train(xy)
        else:
            return self._forward_flux_eval(xy)
    
    def _forward_flux_train(self, xy: torch.Tensor) -> torch.Tensor:
        out = self.forward(xy)
        return out["f"]
    
    @torch.no_grad()
    def _forward_flux_eval(self, xy: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None:
            return self._eval_flux_chunked(xy)
        else:
            out = self.forward(xy)
            return out["f"]

    def _eval_flux_chunked(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the flux on a batch of positions by splitting the batch into
        smaller chunks.
        """
        f_chunks = []
        num_chunks = ceiled_div(len(xy), self.chunk_size)

        progress_bar = self.chunk_progress_bar
        iterator = tqdm.trange(num_chunks) if progress_bar else range(num_chunks)

        chunks = iter_chunks(len(xy), self.chunk_size)
        
        for _, (chunk_start, chunk_stop) in zip(iterator, chunks):
            xy_chunk = xy[chunk_start:chunk_stop]
            out_chunk = self.forward(xy_chunk)
            f_chunks.append(out_chunk["f"])
            
            if progress_bar:
                iterator.set_postfix_str(f"flux_evals:{chunk_stop}/{len(xy)}")
        
        return torch.cat(f_chunks, dim=0)
    
    @normalize_batch_dims(xy=1)
    def total_energy_flux(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the total energy flux at xy.
        
        Args:
            xy (torch.Tensor): Tensor of shape (n, 2) in South-East coordinates.
        
        Returns:
            torch.Tensor: Total energy flux tensor of shape (n,).
        """
        f = self.flux(xy)
        q0 = phy.compute_total_energy_flux(f, self.energy_bins)
        return q0
    
    @normalize_batch_dims(xy=1)
    def mean_energy(self, xy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the mean energy at xy.
        
        Args:
            xy (torch.Tensor): Tensor of shape (n, 2) in South-East coordinates.
        
        Returns:
            torch.Tensor: Mean tensor of shape (n,).
        """
        f = self.flux(xy)
        q0 = phy.compute_mean_energy(f, self.energy_bins)
        return q0

    @normalize_batch_dims(p_frame=1)
    def get_emission_rate(self, p_frame: torch.Tensor, wl: Optional[str] = None) -> torch.Tensor:
        """
        Calculate the emission rate at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in oblique coordinates.
            wl (str, optional): Wavelength label.
        
        Returns:
            torch.Tensor: Emission rate tensor of shape (n,) at the points p_frame
            for the wavelength wl.
        """
        xy, z = p_frame[...,:2], p_frame[...,2]
        f = self.flux(xy)
        emis_mat = self.emis_mats[self.wl_to_idx[wl]]
        return phy.compute_emission_rate(z, f, emis_mat, self.z_bins)
    
    @normalize_batch_dims(p_frame=1)
    def get_electron_density(self, p_frame: torch.Tensor) -> torch.Tensor:
        """
        Calculate the electron density at points p.
        
        Args:
            p_frame (torch.Tensor): Tensor of shape (n, 3) in oblique coordinates.
        
        Returns:
            torch.Tensor: Electron density tensor at the points p of shape (n,).
        """
        xy, z = p_frame[...,:2], p_frame[...,2]
        f = self.flux(xy)
        return phy.compute_electron_density(z, f, self.dens_mat, self.z_bins)

    @overload
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor,
            sampler: RaySampler,
            wl: Optional[str] = None
        ) -> torch.Tensor:
        """
        Integrate the emission along rays. Computes the intersection distances
        with the model's bounding box.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in oblique coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in oblique coordinates.
            sampler (RaySampler): sampler to use for ray integration.
            wl (str, optional): Wavelength label.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
    @overload
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            t: torch.Tensor,
            wl: Optional[str] = None
        ) -> torch.Tensor:
        """
        Integrate the emission along rays.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in oblique coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in oblique coordinates.
            t (torch.Tensor): (n, n_sample) distances to sample points along the rays.
            wl (str, optional): Wavelength label.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
    @overload
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            tn: torch.Tensor,
            tf: torch.Tensor,
            sampler: RaySampler,
            wl: Optional[str] = None
        ) -> torch.Tensor:
        """
        Integrate the emission along rays.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in oblique coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in oblique coordinates.
            tn (torch.Tensor): (n,) near distances of the rays.
            tf (torch.Tensor): (n,) far distances of the rays.
            sampler (RaySampler): sampler to use for ray integration.
            wl (str, optional): Wavelength label.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            t_tn_sampler: Union[torch.Tensor, RaySampler],
            tf_wl: Union[Optional[torch.Tensor], Optional[str]] = None,
            sampler: Optional[RaySampler] = None,
            wl: Optional[str] = None,
        ) -> torch.Tensor:
        
        ro, outer_shape = flatten_batch_dims(ro, 1)
        rd, _ = flatten_batch_dims(rd, 1)
        
        if isinstance(t_tn_sampler, RaySampler):
            # ro, rd, sampler, wl
            tn, tf = self.bbox.intersection(ro, rd)
            wl = tf_wl
            sampler = t_tn_sampler
            p_frame, t = sampler(ro, rd, tn, tf)
        elif tf_wl is None or sampler is None:
            # ro, rd, t, wl
            t = t_tn_sampler
            t, _ = flatten_batch_dims(t, 1)
            wl = tf_wl
            p_frame = gmt.get_ray_points(ro, rd, t)
        else:
            # ro, rd, tn, tf, sampler, wl
            tn = t_tn_sampler
            tf = tf_wl
            tn, _ = flatten_batch_dims(tn, 0)
            tf, _ = flatten_batch_dims(tf, 0)
            p_frame, t = sampler(ro, rd, tn, tf)
        
        xy, z = p_frame[...,:2], p_frame[...,2]
        f = self.flux(xy)
        emis_mat = self.emis_mats[self.wl_to_idx[wl]]
        l = phy.compute_emission_rate(z, f, emis_mat, self.z_bins)
        g = phy.integrate_emis_to_rayleigh(t, l)

        return unflatten_batch_dims(g, outer_shape)
    
    def compute_electron_density(
            self,
            z: torch.Tensor,
            f: torch.Tensor,
        ) -> torch.Tensor:
        """
        Calculate the electron density based on altitude and flux.
        
        Args:
            z (torch.Tensor): Altitude tensor of shape (n,) in oblique frame [km].
            f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
            
        Returns:
            torch.Tensor: Electron density tensor of shape (n,) [cm-3].
        """
        return phy.compute_electron_density(z, f, self.dens_mat, self.z_bins)
    
    def compute_emission_rate(
            self,
            z: torch.Tensor,
            f: torch.Tensor,
            wl: Optional[str] = None
        ) -> torch.Tensor:
        """
        Calculate the emission rate based on altitude and flux.
        
        Args:
            z (torch.Tensor): Altitude tensor of shape (n,) in oblique frame [km].
            f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
            wl (str, optional): Wavelength label.
        
        Returns:
            torch.Tensor: Volume emission rate tensor of shape (n,) [cm-3 s-1].
        """
        emis_mat = self.emis_mats[self.wl_to_idx[wl]]
        return phy.compute_emission_rate(z, f, emis_mat, self.z_bins)
    
    def compute_total_energy_flux(
            self,
            f: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate the total energy flux from the flux tensor.
        
        Args:
            f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        
        Returns:
            torch.Tensor: Total energy flux tensor of shape (n) [W m-2].
        """
        return phy.compute_total_energy_flux(f, self.energy_bins)
    
    def compute_mean_energy(
            self,
            f: torch.Tensor
        ) -> torch.Tensor:
        """
        Calculate the mean energy from the flux tensor.
        
        Args:
            f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        
        Returns:
            torch.Tensor: Mean energy tensor of shape (n) [W m-2].
        """
        return phy.compute_mean_energy(f, self.energy_bins)
    
    @overload
    def generate_image(self, cam: Camera, num_samples: int, nan=0.0, ignore_bbox=False):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            num_samples (int): Number of samples to use for uniform ray integration.
            nan (float): Value to replace NaN values in the image.
            ignore_bbox (bool): Whether to evaluate the model of all camera rays
                regardless if they interesct the bouding box or not
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
    @overload
    def generate_image(self, cam: Camera, sampler: RaySampler, nan=0.0, ignore_bbox=False):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            sampler (RaySampler): Sampler to use for ray integration.
            nan (float): Value to replace NaN values in the image.
            ignore_bbox (bool): Whether to evaluate the model of all camera rays
                regardless if they interesct the bouding box or not
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
    @torch.no_grad()
    def generate_image(self, cam: Camera, n_or_sampler: Union[int, RaySampler], nan=0.0, ignore_bbox=False):
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.frame.from_ecef(ro, is_point=True) # (n, 3)
        rd = self.frame.from_ecef(rd, is_point=False) # (n, 3)
        
        if ignore_bbox:
            box_min = torch.tensor([-torch.inf, -torch.inf, self.bbox.z_min], device=self.device)
            box_max = torch.tensor([ torch.inf,  torch.inf, self.bbox.z_max], device=self.device)
            tn, tf = gmt.ray_box_intersection(ro, rd, box_min, box_max)
        else:
            tn, tf = self.bbox.intersection(ro, rd)
        
        if isinstance(n_or_sampler, RaySampler):
            sampler = n_or_sampler
        else:
            num_samples = n_or_sampler
            sampler = EqualSampler(num_samples)
        
        g = self.integrate_emis_along_ray(ro, rd, tn, tf, sampler, cam.wavelength)
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

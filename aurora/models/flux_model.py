from abc import ABC, abstractmethod
from typing import overload, Optional, Union

import torch
import torch.nn as nn
import tqdm

from aurora.camera import Camera
from aurora.frame import Frame
from aurora.bbox import BBox
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
        return self.forward(xy)
    
    @torch.no_grad()
    def _forward_flux_eval(self, xy: torch.Tensor) -> torch.Tensor:
        if self.chunk_size is not None:
            return self._eval_flux_chunked(xy)
        else:
            return self.forward(xy)

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
            f_chunk = self.forward(xy_chunk)
            f_chunks.append(f_chunk)
            
            if progress_bar:
                iterator.set_postfix_str(f"flux_evals: {chunk_stop}/{len(xy)}")
        
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

    @normalize_batch_dims(p_frame=1)
    def get_emission_rate(self, p_frame: torch.Tensor) -> torch.Tensor:
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
        return phy.compute_emission_rate(z, f, self.emis_mat, self.altitude_bins)
    
    @normalize_batch_dims(p_frame=1)
    def get_electron_density(self, p_frame: torch.Tensor) -> torch.Tensor:
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
        return phy.compute_electron_density(z, f, self.dens_mat, self.altitude_bins)

    @overload
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor,
            sampler: RaySampler
        ) -> torch.Tensor:
        """
        Integrate the emission along rays. Computes the intersection distances
        with the model's bounding box.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in frame coordinates.
            sampler (RaySampler): sampler to use for ray integration.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
    @overload
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            t: torch.Tensor
        ) -> torch.Tensor:
        """
        Integrate the emission along rays.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in frame coordinates.
            t (torch.Tensor): (n, n_sample) distances to sample points along the rays.
        
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
            sampler: RaySampler
        ) -> torch.Tensor:
        """
        Integrate the emission along rays.
        
        Args:
            ro (torch.Tensor): Ray origins of shape (n, 3) in frame coordinates.
            rd (torch.Tensor): Ray directions of shape (n, 3) in frame coordinates.
            tn (torch.Tensor): (n,) near distances of the rays.
            tf (torch.Tensor): (n,) far distances of the rays.
            sampler (RaySampler): sampler to use for ray integration.
        
        Returns:
            torch.Tensor: Integrated emission tensor of shape (n,).
        """
    def integrate_emis_along_ray(
            self, 
            ro: torch.Tensor, 
            rd: torch.Tensor, 
            t_tn_sampler: Union[torch.Tensor, RaySampler],
            tf: Optional[torch.Tensor] = None,
            sampler: Optional[RaySampler] = None,
        ) -> torch.Tensor:
        
        ro, outer_shape = flatten_batch_dims(ro, 1)
        rd, _ = flatten_batch_dims(rd, 1)
        
        if isinstance(t_tn_sampler, RaySampler):
            # ro, rd, sampler
            tn, tf = self.bbox.intersection(ro, rd)
            sampler = t_tn_sampler
            p_frame, t = sampler(ro, rd, tn, tf)
        elif tf is None or sampler is None:
            # ro, rd, t
            t = t_tn_sampler
            t, _ = flatten_batch_dims(t, 1)
            p_frame = gmt.get_ray_points(ro, rd, t)
        else:
            # ro, rd, tn, tf, sampler
            tn = t_tn_sampler
            tn, _ = flatten_batch_dims(tn, 0)
            tf, _ = flatten_batch_dims(tf, 0)
            p_frame, t = sampler(ro, rd, tn, tf)
        
        p_enu = self.frame.to_local_enu(p_frame, is_point=True)
        xy, z = p_frame[...,:2], p_enu[...,2]
        f = self.flux(xy)
        l = phy.compute_emission_rate(z, f, self.emis_mat, self.altitude_bins)
        g = phy.integrate_emis_to_rayleigh(t, l)

        return unflatten_batch_dims(g, outer_shape)
    
    @overload
    def generate_image(self, cam: Camera, num_samples: int, nan=0.0):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            num_samples (int): Number of samples to use for ray integration.
            nan (float): Value to replace NaN values in the image.
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
    @overload
    def generate_image(self, cam: Camera, sampler: RaySampler, nan=0.0):
        """
        Generate an image from the camera using the integrated emission along rays.
        
        Args:
            cam (Camera): Camera object to generate the image from.
            sampler (RaySampler): Sampler to use for ray integration.
            nan (float): Value to replace NaN values in the image.
        
        Returns:
            torch.Tensor: Image tensor of shape (cam.height, cam.width).
        """
    @torch.no_grad()
    def generate_image(self, cam: Camera, n_or_sampler: Union[int, RaySampler], nan=0.0):
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.frame.from_ecef(ro, is_point=True) # (n, 3)
        rd = self.frame.from_ecef(rd, is_point=False) # (n, 3)
        
        tn, tf = self.bbox.intersection(ro, rd)
        
        if isinstance(n_or_sampler, RaySampler):
            sampler = n_or_sampler
        else:
            num_samples = n_or_sampler
            sampler = EqualSampler(num_samples)
        
        g = self.integrate_emis_along_ray(ro, rd, tn, tf, sampler)
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

from abc import ABC, abstractmethod
from typing import overload, Optional, Union, Dict

import torch
import torch.nn as nn
import tqdm

from aurora.camera import Camera
from aurora.bbox import BBox
import aurora.geometry as gmt
import aurora.physics as phy
from aurora.utils import (
    normalize_batch_dims,
    iter_chunks,
    flatten_batch_dims,
    unflatten_batch_dims
)
from aurora.samplers import RaySampler, EqualSampler


class ModelConfig:
    def __init__(
        self,
        bbox: BBox,
        altitude_bins: torch.Tensor,
        energy_bins: torch.Tensor,
        emis_mat: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        dens_mat: Optional[torch.Tensor] = None,
    ):
        if emis_mat is None and dens_mat is None:
            raise ValueError("Missing emission or density matrix.")
        
        if emis_mat is not None:
            if torch.is_tensor(emis_mat):
                # Single wavelength
                self.emis_mats = emis_mat.unsqueeze(0) # (1, n_z, n_E)
                self.wl_to_idx = {None: 0}
            else:
                # Multiple wavelengths
                self.wl_to_idx = {}
                emis_mats_list = []
                for i, (wl, mat) in enumerate(emis_mat.items()):
                    emis_mats_list.append(mat)
                    self.wl_to_idx[wl] = i
                self.emis_mats = torch.stack(emis_mats_list) # (n_lam, n_z, n_E)
        else:
            self.emis_mats = None
            self.wl_to_idx = None
        
        self.dens_mat = dens_mat
        
        self.altitude_bins = altitude_bins
        self.energy_bins = energy_bins

        self.bbox = bbox


class FluxModel(nn.Module, ABC):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.bbox = config.bbox
        
        self.register_buffer("emis_mats", config.emis_mats)
        self.register_buffer("dens_mat", config.dens_mat)
        self.register_buffer("altitude_bins", config.altitude_bins)
        self.register_buffer("energy_bins", config.energy_bins)
        self.wl_to_idx = config.wl_to_idx

        # Compute the altitude bin edges in oblique frame
        z_bins = self.frame.altitude_to_z(config.altitude_bins)
        self.register_buffer("z_bins", z_bins)

        self.num_bins = len(config.energy_bins) - 1
    
    @property
    def device(self):
        return self.altitude_bins.device
    
    @property
    def frame(self):
        return self.bbox.frame
    
    @property
    def is_multi_wavelength(self):
        return None not in self.wl_to_idx
    
    @property
    def wavelengths(self):
        return tuple(self.wl_to_idx.keys())
    
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
        return self.forward(xy)["f"]
    
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
    def generate_image(
        self,
        cam: Camera,
        num_samples: int,
        nan=0.0,
        ignore_bbox=False,
        chunk_size=16384,
        progress_bar=False
    ):
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
    def generate_image(
        self,
        cam: Camera,
        sampler: RaySampler,
        nan=0.0,
        ignore_bbox=False,
        chunk_size=16384,
        progress_bar=False
    ):
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
    def generate_image(
        self,
        cam: Camera,
        n_or_sampler: Union[int, RaySampler],
        nan=0.0,
        ignore_bbox=False,
        chunk_size=16384,
        progress_bar=False
    ):
        ro, rd = cam.create_rays_ecef(self.device)
        ro = self.bbox.frame.from_ecef(ro, is_point=True) # (n, 3)
        rd = self.bbox.frame.from_ecef(rd, is_point=False) # (n, 3)
        
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
        
        chunks = list(iter_chunks(len(ro), chunk_size))
        it = tqdm.tqdm(chunks) if progress_bar else chunks

        gs = []
        for (beg, end) in it:
            ro_ = ro[beg:end, ...]
            rd_ = rd[beg:end, ...]
            tn_ = tn[beg:end, ...]
            tf_ = tf[beg:end, ...]
            g_ = self.integrate_emis_along_ray(ro_, rd_, tn_, tf_, sampler, cam.wavelength)
            gs.append(g_)
            
            if progress_bar:
                it.set_postfix_str(f"pixels:{end}/{len(ro)}")
        
        g = torch.cat(gs, dim=0)
        img = g.reshape_as(cam.image)
        img = torch.nan_to_num(img, nan=nan)
        return img

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.bbox.to(*args, **kwargs)
        return self
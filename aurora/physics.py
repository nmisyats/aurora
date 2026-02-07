from typing import Union, Dict, Optional

import torch


class PhysicalModel:
    """Container to group aurora physics tensors.
    
    Attributes:
        emis_mats (torch.Tensor): Tensor of shape (n_lam, n_z, n_E) storing
            emission matrices for each of the n_lam wavelengths.
        dens_mat (torch.Tensor): Tensor of shape (n_z, n_E) storing the
            density matrix.
        altitude_bins (torch.Tensor): Tensor of shape (n_z+1,) storing the edges
            of each altitude bin using by the emission and density matrices.
        energy_bins (torch.Tensor): Tensor of shape (n_E+1,) storing the edges
            of each energy bin using by the emission and density matrices.
        wl_to_idx (dict[str, int]): `wl_to_idx[wl]` is the index `i` of the
            emission matrix `emis_mat[i]` for the wavelength `wl`.
    """
    
    def __init__(
        self,
        altitude_bins: torch.Tensor,
        energy_bins: torch.Tensor,
        emis_mats: Optional[Union[torch.Tensor, Dict[str, torch.Tensor]]] = None,
        dens_mat: Optional[torch.Tensor] = None,
    ):
        if emis_mats is None and dens_mat is None:
            raise ValueError("Missing emission or density matrix.")
        
        if emis_mats is not None:
            if torch.is_tensor(emis_mats):
                # Single wavelength
                self.emis_mats = emis_mats.unsqueeze(0) # (1, n_z, n_E)
                self.wl_to_idx = {None: 0}
            else:
                # Multiple wavelengths
                self.wl_to_idx = {}
                emis_mats_list = []
                for i, (wl, emis_mat) in enumerate(emis_mats.items()):
                    emis_mats_list.append(emis_mat)
                    self.wl_to_idx[wl] = i
                self.emis_mats = torch.stack(emis_mats_list) # (n_lam, n_z, n_E)
        else:
            self.emis_mats = None
            self.wl_to_idx = None
        
        self.dens_mat = dens_mat
        
        self.altitude_bins = altitude_bins
        self.energy_bins = energy_bins
    
    def to(self, device: torch.device):
        """Move all tensors to the specified device and return a new
        PhysicalModel instance."""
        new_physics = object.__new__(PhysicalModel)

        if self.emis_mats is not None:
            new_physics.emis_mats = self.emis_mats.to(device)
        else:
            new_physics.emis_mats = None
        
        self.wl_to_idx = self.wl_to_idx
        
        if self.dens_mat is not None:
            new_physics.dens_mat = self.dens_mat.to(device)
        else:
            new_physics.dens_mat = None
        
        new_physics.altitude_bins = self.altitude_bins
        new_physics.energy_bins = self.energy_bins
        
        return new_physics


def compute_emission_rate(
        z: torch.Tensor,
        f: torch.Tensor,
        emis_mat: torch.Tensor,
        altitude_bins: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the emission rate based on altitude and flux.
    
    Args:
        z (torch.Tensor): Altitude tensor of shape (n,) [km].
        f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        emis_mat (torch.Tensor): Emission matrix of shape (n_z, n_E) [TODO].
        altitude_bins (torch.Tensor): Edges of altitude bins of shape (n_z+1,) [km].
    
    Returns:
        torch.Tensor: Volume emission rate tensor of shape (n,) [cm-3 s-1].
    """
    z_idx = torch.bucketize(z.contiguous(), altitude_bins) - 1
    z_idx = torch.clamp(z_idx, 0, emis_mat.shape[1]-1)
    m_z = emis_mat[z_idx.flatten(),:]
    m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
    l = torch.sum(m_z * f, dim=-1)
    return l

def compute_electron_density(
        z: torch.Tensor,
        f: torch.Tensor,
        dens_mat: torch.Tensor,
        altitude_bins: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the electron density based on altitude and flux.
    
    Args:
        z (torch.Tensor): Altitude tensor of shape (n,) [km].
        f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        dens_mat (torch.Tensor): Density matrix of shape (n_z, n_E) [TODO].
        altitude_bins (torch.Tensor): Edges of altitude bins of shape (n_z+1,) [km].
        
    Returns:
        torch.Tensor: Electron density tensor of shape (n,) [cm-3].
    """
    z_idx = torch.bucketize(z.contiguous(), altitude_bins) - 1
    z_idx = torch.clamp(z_idx, 0, dens_mat.shape[1]-1)
    m_z = dens_mat[z_idx.flatten(),:]
    m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
    d2 = torch.sum(m_z * f, dim=-1)
    d = torch.sqrt(d2)
    return d

def integrate_emis_to_rayleigh(
        t: torch.Tensor,
        l: torch.Tensor,
    ) -> torch.Tensor:
    """
    Integrate the emission rate over sample points to compute the gray level in units
    of Rayleigh.
    
    Args:
        t (torch.Tensor): Distances from view point of each emission sample (n, n_samples) [km].
        l (torch.Tensor): Emission rate at each sampled distance (n, n_samples) [cm-3 s-1].
    
    Returns:
        torch.Tensor: Rayleigh gray level tensor of shape (n,) [Rayleigh].
    """
    g = torch.trapezoid(l, t)
    g /= 10.0 # scale to Rayleigh units
    return g

def compute_total_energy_flux(
        f: torch.Tensor,
        energy_bins: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the total energy flux from the flux tensor.
    
    Args:
        f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        energy_bins (torch.Tensor): Edges of energy bins of shape (n_E+1,) [eV].
    
    Returns:
        torch.Tensor: Total energy flux tensor of shape (n) [mW m-2].
    """
    e = 1.602e-19
    lower_E, upper_E = energy_bins[:-1], energy_bins[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    q = (10**3) * e * (10**4) * torch.pi * (f * E * dE)
    q = torch.sum(q, dim=-1)
    return q

def compute_mean_energy(
        f: torch.Tensor,
        energy_bins: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the mean energy from the flux tensor.
    
    Args:
        f (torch.Tensor): Flux tensor of shape (n, n_E) [cm-2 s-1 eV-1].
        energy_bins (torch.Tensor): Edges of energy bins of shape (n_E+1,) [eV].
    
    Returns:
        torch.Tensor: Mean energy tensor of shape (n,) [keV].
    """
    e = 1.602e-19
    lower_E, upper_E = energy_bins[:-1], energy_bins[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    eflux = torch.pi * (f * E * dE)
    eflux = torch.sum(eflux, dim=-1)
    nflux = torch.pi * (f * dE)
    nflux = torch.sum(nflux, dim=-1)
    emean = eflux / nflux * 1e-3
    return emean
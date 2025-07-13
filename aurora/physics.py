from dataclasses import dataclass

import torch


@dataclass
class PhysicalModel:
    """Container to group aurora physics tensors."""    
    emis_mat: torch.Tensor
    dens_mat: torch.Tensor
    altitude_bins: torch.Tensor
    energy_bins: torch.Tensor

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
        torch.Tensor: Total energy flux tensor of shape (n) [W m-2].
    """
    e = 1.602e-19
    lower_E, upper_E = energy_bins[:-1], energy_bins[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    q = (10**3) * e * (10**4) * torch.pi * (f * E * dE)
    q = torch.sum(q, dim=-1)
    return q

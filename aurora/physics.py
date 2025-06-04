from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def emis_rate(
        z: torch.Tensor,
        f: torch.Tensor,
        M_emis: torch.Tensor,
        z_edges: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the emission rate based on altitude and flux.
    
    Args:
        z (torch.Tensor): Altitude tensor of shape (...,) [km].
        f (torch.Tensor): Flux tensor of shape (..., n_E) [cm-2 s-1 eV-1].
        M_emis (torch.Tensor): Emission matrix of shape (n_z, n_E) [TODO].
        z_edges (torch.Tensor): Edges of altitude bins of shape (n_z+1,) [km].
    
    Returns:
        torch.Tensor: Voume emission rate tensor of shape (...,) [cm-3 s-1].
    """
    z_idx = torch.bucketize(z.contiguous(), z_edges) - 1
    z_idx = torch.clamp(z_idx, 0, M_emis.shape[1]-1)
    m_z = M_emis[z_idx.flatten(),:]
    m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
    l = torch.sum(m_z * f, dim=-1)
    return l

def elec_dens(
        z: torch.Tensor,
        f: torch.Tensor,
        M_dens: torch.Tensor,
        z_edges: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the electron density based on altitude and flux.
    
    Args:
        z (torch.Tensor): Altitude tensor of shape (...,) [km].
        f (torch.Tensor): Flux tensor of shape (..., n_E) [cm-2 s-1 eV-1].
        M_dens (torch.Tensor): Density matrix of shape (n_z, n_E) [TODO].
        z_edges (torch.Tensor): Edges of altitude bins of shape (n_z+1,) [km].
        
    Returns:
        torch.Tensor: Electron density tensor of shape (...,) [cm-3].
    """
    z_idx = torch.bucketize(z.contiguous(), z_edges) - 1
    z_idx = torch.clamp(z_idx, 0, M_dens.shape[1]-1)
    m_z = M_dens[z_idx.flatten(),:]
    m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
    d2 = torch.sum(m_z * f, dim=-1)
    d = torch.sqrt(d2)
    return d

def int_emis_rayleigh(
        t: torch.Tensor,
        l: torch.Tensor,
    ) -> torch.Tensor:
    """
    Integrate the emission rate over a sample points to compute the gray level in units
    of Rayleigh.
    
    Args:
        t (torch.Tensor): Distances from view point of each emission sample (...,) [km].
        l (torch.Tensor): Emission rate at each sampled distance (..., 3) [cm-3 s-1].
    
    Returns:
        torch.Tensor: Rayleigh gray level tensor of shape (...,) [Rayleigh].
    """
    g = torch.trapezoid(l, t)
    g /= 10.0 # scale to Rayleigh units
    return g

def total_energy_flux(
        f: torch.Tensor,
        E_edges: torch.Tensor
    ) -> torch.Tensor:
    """
    Calculate the total energy flux from the flux tensor.
    
    Args:
        f (torch.Tensor): Flux tensor of shape (..., n_E) [cm-2 s-1 eV-1].
        E_edges (torch.Tensor): Edges of energy bins of shape (n_E+1,) [eV].
    
    Returns:
        torch.Tensor: Total energy flux tensor of shape (...) [W m-2].
    """
    e = 1.602e-19
    lower_E, upper_E = E_edges[:-1], E_edges[1:]
    E = (lower_E + upper_E) / 2.0
    dE = upper_E - lower_E
    q = (10**3) * e * (10**4) * torch.pi * (f * E * dE)
    q = torch.sum(q, dim=-1)
    return q

# class PhysicalModel:
#     def __init__(
#             self,
#             altitude_bins: torch.Tensor,
#             energy_bins: torch.Tensor,
#             emission_matrix: torch.Tensor,
#             density_matrix: torch.Tensor,
#             device: torch.device
#         ):
#         self.device = device
#         self.z_edges = altitude_bins.to(self.device)
#         self.M_emis = emission_matrix.T.to(self.device)
#         self.M_dens = density_matrix.T.to(self.device).square()
#         self.E_edges = energy_bins.to(self.device)

#     def emis_rate(self, p_une: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
#         z_une = p_une[..., 0]
#         z_idx = torch.bucketize(z_une.contiguous(), self.z_edges) - 1
#         z_idx = torch.clamp(z_idx, 0, self.M_emis.shape[1]-1)
#         m_z = self.M_emis[z_idx.flatten(),:]
#         m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
#         l = torch.sum(m_z * f, dim=-1)
#         return l
    
#     def e_dens(self, p_une: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
#         z_une = p_une[..., 0]
#         z_idx = torch.bucketize(z_une.contiguous(), self.z_edges) - 1
#         z_idx = torch.clamp(z_idx, 0, self.M_dens.shape[1]-1)
#         m_z = self.M_dens[z_idx.flatten(),:]
#         m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
#         d2 = torch.sum(m_z * f, dim=-1)
#         d = torch.sqrt(d2)
#         return d

#     def integrate_gray_level(
#             self,
#             p: torch.Tensor,
#             t: torch.Tensor,
#             f: torch.Tensor
#         ) -> torch.Tensor:
#         l = self.emis_rate(p, f)
#         g = torch.trapezoid(l, t)
#         g /= 10.0
#         return g
    
#     def total_energy_flux(self, f: torch.Tensor) -> torch.Tensor:
#         e = 1.602e-19
#         lower_E, upper_E = self.E_edges[:-1], self.E_edges[1:]
#         E = (lower_E + upper_E) / 2.0
#         dE = upper_E - lower_E
#         q = (10**3) * e * (10**4) * torch.pi * (f * E * dE)
#         q = torch.sum(q, dim=-1)
#         return q


class ElectronFluxModel(ABC):
    @abstractmethod
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        ...

class TrainableFluxModel(ElectronFluxModel, nn.Module):
    def flux(self, xy: torch.Tensor):
        orig_shape = xy.shape[:-1] # (k1, k2, ..., kn, 2)
        xy = xy.reshape(-1, 2) # (N, 2)
        f = self.forward(xy) # (N, n_bins)
        n_bins = f.shape[-1]
        return f.reshape(*orig_shape, n_bins)

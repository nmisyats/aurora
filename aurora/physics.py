import torch


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

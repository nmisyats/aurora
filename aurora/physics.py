from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from aurora.frame import ReferenceFrame


class PhysicalModel:
    def __init__(
            self,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            emission_matrix: torch.Tensor,
            frame: ReferenceFrame,
            device: torch.device
        ):
        self.device = device
        self.frame = frame
        self.z_edges = altitude_bins.to(self.device)
        self.m_mat = emission_matrix.to(self.device)
        self.E_edges = energy_bins.to(self.device)

    def L(self, z: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        z = z + self.frame.origin_altitude
        z_idx = torch.bucketize(z.contiguous(), self.z_edges) - 1
        z_idx = torch.clamp(z_idx, 0, self.m_mat.shape[1]-1)
        m_z = self.m_mat[z_idx.flatten(),:]
        m_z = m_z.reshape(*z_idx.shape, m_z.shape[-1])
        l = torch.sum(m_z * f, dim=-1)
        return l
    
    def integrate_g(self,
          rd: torch.Tensor,
          t: torch.Tensor,
          p: torch.Tensor,
          f: torch.Tensor
        ) -> torch.Tensor:
        z = p[:, 2]
        l = self.L(z, f)
        g = torch.trapezoid(l, t)
        g *= torch.sqrt(rd @ self.frame.metric_tensor @ rd)
        g /= 10.0
        return g
    
    def q0(self, f: torch.Tensor) -> torch.Tensor:
        e = 1.602e-19
        lower_E, upper_E = self.E_edges[:-1], self.E_edges[1:]
        E = (lower_E + upper_E) / 2.0
        dE = upper_E - lower_E
        q = (10**3) * e * (10**4) * torch.pi * (f * E * dE)
        q = torch.sum(q, dim=-1)
        return q


class ElectronFluxModel(ABC):
    @abstractmethod
    def f_at(self, xy: torch.Tensor) -> torch.Tensor:
        ...

class TrainableFluxModel(ElectronFluxModel, nn.Module):
    def f_at(self, xy: torch.Tensor):
        orig_shape = xy.shape[:-1] # (k1, k2, ..., kn, 2)
        xy = xy.reshape(-1, 2) # (N, 2)
        f = self.forward(xy) # (N, n_bins)
        n_bins = f.shape[-1]
        return f.reshape(*orig_shape, n_bins)

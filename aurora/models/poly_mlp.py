from typing import Literal

import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class PolyMLP(FluxModel):
    """MLP learning a polynomial basis of the flux"""
    def __init__(
            self,
            config: ModelConfig,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            basis_fn: Literal["mono", "chebyshev"] = "mono",
            num_basis: int = 8,
            num_hidden: int = 4,
            hidden_size: int =  128
        ):
        super().__init__(config)

        self.num_basis = num_basis
        self.max_log_flux = max_log_flux

        self.encoder = ann.FourierEncoder(encoding_exp)

        encode_dim = self.encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = ann.MLP(encode_dim, self.num_basis, hidden_sizes)
        
        # Pre-compute basis functions
        self.register_buffer('basis_functions', self._create_basis_functions(basis_fn))

    def _create_basis_functions(self, basis_fn: str):
        # Work in normalized log-energy space
        log_energies = torch.log(self.energy_bins)
        log_e_min = log_energies.min()
        log_e_max = log_energies.max()
        log_e_norm = (log_energies - log_e_min) / (log_e_max - log_e_min) # [0, 1]
        log_e_norm = 2.0 * log_e_norm - 1.0 # [-1, 1]
        # Create basis functions
        if basis_fn == "mono":
            return self._create_monomial_basis(log_e_norm)
        elif basis_fn == "chebyshev":
            return self._create_chebyshev_basis(log_e_norm)
        raise ValueError(f"Unsupported basis function \"{basis_fn}\"")
    
    def _create_monomial_basis(self, x: torch.Tensor):
        basis = torch.zeros(self.num_basis, len(x), device=x.device)
        for i in range(self.num_basis):
            basis[i] = x ** i
        basis = (basis + 1.0) / 2.0
        return basis # (num_basis, num_edges)
    
    def _create_chebyshev_basis(self, x: torch.Tensor):
        basis = torch.zeros(self.num_basis, len(x), device=x.device)
        basis[0] = torch.ones_like(x)
        basis[1] = x
        for i in range(2, self.num_basis):
            basis[i] = 2.0 * x * basis[i-1] - basis[i-2]
        basis = (basis + 1.0) / 2.0
        return basis # (num_basis, num_edges)
    
    def forward(self, xy: torch.Tensor):
        xy = self.bbox.normalize_xy(xy)
        xy_enc = self.encoder(xy)
        coeffs = self.mlp(xy_enc) # (batch_size, num_basis)
        log_f_at_edges = torch.matmul(coeffs, self.basis_functions) # (batch_size, num_edges)
        log_f_at_edges *= self.max_log_flux
        f_at_edges = torch.pow(10.0, log_f_at_edges)
        f = 0.5 * (f_at_edges[:, :-1] + f_at_edges[:, 1:])
        f = torch.max(f, torch.tensor(1e-3, device=f.device))
        log_f = torch.log(f)
        return {
            "f": f,
            "f_at_edges": f_at_edges,
            "log_f_at_edges": log_f_at_edges,
            "log_f": log_f,
            "coeffs": coeffs
        }

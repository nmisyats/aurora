import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class PolyMLP(FluxModel):
    """MLP learning a polynomial basis of the flux"""
    def __init__(
            self,
            frame: Frame,
            bbox: BBox,
            emis_mat: torch.Tensor,
            dens_mat: torch.Tensor,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            num_basis: int = 8,
            num_hidden: int = 4,
            hidden_size: int =  128
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.num_basis = num_basis
        self.max_log_flux = max_log_flux

        self.fourier_encoder = ann.FourierEncoder(encoding_exp)

        encode_dim = self.fourier_encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = ann.create_mlp(encode_dim, *hidden_sizes, self.num_bins)
        
        # Pre-compute basis functions
        self.register_buffer('basis_functions', self._create_basis_functions())
    
    def _create_basis_functions(self):
        # Work in normalized log-energy space
        e_norm = self._normalize_energy(self.energy_bins) + 1e-8
        log_energies = torch.log(e_norm)
        log_e_min = log_energies.min()
        log_e_max = log_energies.max()
        log_e_norm = (log_energies - log_e_min) / (log_e_max - log_e_min)
        # Create basis functions
        basis = torch.zeros(self.num_basis, len(self.energy_bins))
        for i in range(self.num_basis):
            basis[i] = log_e_norm ** i
        return basis # (num_basis, num_edges)
    
    def forward(self, xy: torch.Tensor):
        xy = self._normalize_xy(xy)
        x = self.fourier_encoder(xy)
        coeffs = self.mlp(x) # (batch_size, num_basis + 1)
        log_f_edges = torch.matmul(coeffs, self.basis_functions) # (batch_size, num_edges)
        log_f = 0.5 * (log_f_edges[:, :-1] + log_f_edges[:, 1:]) # (batch_size, num_bins)
        f = ann.clamped_exp10(log_f, 0.0, self.max_log_flux)
        return {"f": f, "log_f": log_f, "coeffs": coeffs}

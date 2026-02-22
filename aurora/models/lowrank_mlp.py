from typing import Optional

import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.dnn import FourierEncoder, MLP


class LowRankMLP(FluxModel):
    """
    Low-rank spectral model:
        log_f(xy, E) = log_f_mean(E) + coeffs(xy) @ basis(E)
    where coeffs(xy) is predicted by an MLP and basis is learned globally.
    """
    def __init__(
            self,
            config: ModelConfig,
            rank: int = 8,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            num_hidden: int = 4,
            hidden_size: int = 128,
            coeff_scale: float = 1.0,
            init_mean: Optional[float] = None
        ):
        super().__init__(config)

        if rank <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.rank = rank
        self.max_log_flux = max_log_flux
        self.coeff_scale = coeff_scale

        self.encoder = FourierEncoder(encoding_exp)
        enc_dim = self.encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = MLP(enc_dim, self.rank, hidden_sizes)

        if init_mean is None:
            init_mean = self.max_log_flux / 2.0

        self.log_f_mean = nn.Parameter(torch.full((self.num_bins,), init_mean))
        self.basis = nn.Parameter(torch.randn(self.rank, self.num_bins) * 1e-2)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, xy: torch.Tensor):
        xy_enc = self.encoder(xy)
        coeffs_raw = self.mlp(xy_enc)
        coeffs = torch.tanh(coeffs_raw) * self.coeff_scale

        log_f = self.log_f_mean.unsqueeze(0) + coeffs @ self.basis
        log_f = torch.clamp_max(log_f, self.max_log_flux)
        f = torch.pow(10.0, log_f)
        return {
            "coeffs": coeffs,
            "basis": self.basis,
            "log_f_mean": self.log_f_mean,
            "log_f": log_f,
            "f": f,
        }


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.dnn import FourierEncoder, MLP


class SpectralMLP(FluxModel):
    """MLP outputing the energy spectrum from the xy position"""
    def __init__(
            self,
            config: ModelConfig,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            num_hidden: int = 4,
            hidden_size: int = 128,
            init_bias: Optional[float] = None
        ):
        super().__init__(config)
        
        self.max_log_flux = max_log_flux
        
        self.encoder = FourierEncoder(encoding_exp)
        enc_dim = self.encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = MLP(enc_dim, self.num_bins, hidden_sizes)

        if init_bias is None:
            init_bias = self.max_log_flux / 2.0
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, xy: torch.Tensor):
        xy_enc = self.encoder(xy)
        log_f = self.mlp(xy_enc)
        log_f = torch.clamp_max(log_f, self.max_log_flux)
        f = torch.pow(10.0, log_f)
        return {"log_f": log_f, "f": f}

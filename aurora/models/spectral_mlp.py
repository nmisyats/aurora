from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class SpectralMLP(FluxModel):
    """MLP outputing the energy spectrum from the xy position"""
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
            num_hidden: int = 4,
            hidden_size: int =  128
        ):
        assert num_hidden > 0
        assert hidden_size > 0

        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(encoding_exp)
        self.max_log_flux = max_log_flux

        encode_dim = self.fourier_encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = ann.create_mlp(encode_dim, *hidden_sizes, self.num_bins)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
    def forward(self, xy: torch.Tensor):
        xy_norm = self._normalize_xy(xy)
        xy_enc = self.fourier_encoder(xy_norm)
        log_f = self.mlp(xy_enc)
        f = ann.clamped_exp10(log_f, 0.0, self.max_log_flux)
        return {"log_f": log_f, "f": f}

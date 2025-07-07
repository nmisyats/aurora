from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class ResidualMLP(FluxModel):
    """Double MLP learning coarse and detailed flux in parallel"""
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
            low_flux_res: int = 8
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(encoding_exp)
        self.max_log_flux = max_log_flux

        encode_dim = self.fourier_encoder.output_dim(2)

        self.coarse_mlp = ann.create_mlp(encode_dim, 128, 64, low_flux_res)
        self.details_mlp = ann.create_mlp(encode_dim + low_flux_res, 32, self.num_bins)
    
    def forward(self, xy: torch.Tensor):
        xy_norm = self._normalize_xy(xy)
        xy_enc = self.fourier_encoder(xy_norm)
        log_f_low = self.coarse_mlp(xy_enc)

        log_f_coarse = F.interpolate(
            input=log_f_low.unsqueeze(1),
            size=self.num_bins,
            mode='linear', 
            align_corners=True
        ).squeeze(1) # (n, num_bins)
        f_coarse = ann.clamped_exp10(log_f_coarse, 0.0, self.max_log_flux)

        details_input = torch.cat([log_f_low, xy_enc], dim=-1)
        log_f_details = self.details_mlp(details_input)
        log_f_details = torch.clamp(log_f_details, 0.5, 2.0)
        log_f_fine = f_coarse + log_f_details
        f_fine = ann.clamped_exp10(log_f_fine, 0.0, self.max_log_flux)

        return {
            "f": f_fine,
            "f_fine": f_fine,
            "f_coarse": f_coarse,
            "log_f_low": log_f_low,
            "log_f_coarse": log_f_coarse,
            "log_f_fine": log_f_fine
        }

import torch
import torch.nn as nn

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
            max_log_flux: float = 7.0
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(encoding_exp)
        self.max_log_flux = max_log_flux

        encode_dim = self.fourier_encoder.output_dim(2)
        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, len(self.energy_bins) - 1)
        )

        self.exp10 = ann.Exponentiate(10.0, 0.0, self.max_log_flux)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
    def forward(self, xy: torch.Tensor):
        x = self._normalize_xy(xy)
        x = self.fourier_encoder(x)
        x = self.mlp(x)
        x = self.exp10(x)
        return x


class SpectralResMLP(FluxModel):
    """Spectral MLP with a residual connection"""
    def __init__(
            self,
            frame: Frame,
            bbox: BBox,
            emis_mat: torch.Tensor,
            dens_mat: torch.Tensor,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(encoding_exp)
        self.max_log_flux = max_log_flux

        encode_dim = self.fourier_encoder.output_dim(2)
        self.mlp1 = nn.Sequential(
            nn.Linear(encode_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128 + encode_dim, 128), nn.ReLU(),
            nn.Linear(128, self.num_bins), nn.Sigmoid()
        )

        self.exp10 = ann.Exponentiate(10.0, 0.0, self.max_log_flux)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
    def forward(self, xy: torch.Tensor):
        xy = self._normalize_xy(xy)
        x = self.fourier_encoder(xy)
        x0 = x
        x = self.mlp1(x)
        x = torch.cat([x, x0], dim=-1)
        x = self.mlp2(x)
        x = self.exp10(x)
        return x
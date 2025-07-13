import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel, ModelConfig
import aurora.dnn as ann


class SpectralMLP(FluxModel):
    """MLP outputing the energy spectrum from the xy position"""
    def __init__(
            self,
            config: ModelConfig,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            num_hidden: int = 4,
            hidden_size: int = 128
        ):
        super().__init__(config)

        self.max_log_flux = max_log_flux

        self.encoder = ann.FourierEncoder(encoding_exp)

        encode_dim = self.encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = ann.create_mlp(encode_dim, *hidden_sizes, self.num_bins)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
    def forward(self, xy: torch.Tensor):
        xy_norm = self.bbox.norm_xy(xy)
        xy_enc = self.encoder(xy_norm)
        log_f = self.mlp(xy_enc)
        f = ann.clamped_exp10(log_f, 0.0, self.max_log_flux)
        return {"log_f": log_f, "f": f}

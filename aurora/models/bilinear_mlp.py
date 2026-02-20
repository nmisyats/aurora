from typing import Optional

import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.dnn import FourierEncoder, MLP


class BilinearMLP(FluxModel):
    """Hybrid model using two MLPs for position and energy embedding with final bilinear combination"""
    def __init__(
            self,
            config: ModelConfig,
            position_embed: int = 8,
            energy_embed: int = 8,
            position_enc: int = 4,
            energy_enc: int = 2,
            max_log_flux: float = 7.0,
            embed_hidden_size: int = 64,
            init_bias: Optional[float] = None
        ):
        super().__init__(config)

        self.max_log_flux = max_log_flux

        self.position_encoder = FourierEncoder(position_enc)
        self.energy_encoder = FourierEncoder(energy_enc)

        position_encode_dim = self.position_encoder.output_dim(2)
        if position_embed:
            position_embed_dim = position_embed
            self.position_embedder = MLP(
                position_encode_dim,
                position_embed_dim,
                (embed_hidden_size,)
            )
        else:
            self.position_embedder = nn.Identity()
            position_embed_dim = position_encode_dim

        energy_encode_dim = self.energy_encoder.output_dim(1)
        if energy_embed:
            energy_embed_dim = energy_embed
            self.energy_embedder = MLP(
                energy_encode_dim,
                energy_embed_dim,
                (embed_hidden_size,)
            )
        else:
            self.energy_embedder = nn.Identity()
            energy_embed_dim = energy_encode_dim

        self.combiner = nn.Bilinear(position_embed_dim, energy_embed_dim, 1)

        if init_bias is None:
            init_bias = self.max_log_flux / 2.0
        nn.init.constant_(self.combiner.bias, init_bias)
    
    def forward(self, xy: torch.Tensor):
        if xy.dim() == 1:
            return self._forward_single(xy)
        else:
            return self._forward_batch(xy)
    
    def _forward_single(self, xy: torch.Tensor):
        # xy: (2,) -> output: (N,)
        out = self._forward_batch(xy.unsqueeze(0))
        return {k: t.squeeze(0) for k, t in out.items()}

    def _forward_batch(self, xy: torch.Tensor):
        # xy: (B, 2)
        B = xy.shape[0]

        xy_encoded = self.position_encoder(xy) # (B, pos_enc)
        xy_embed = self.position_embedder(xy_encoded) # (B, pos_embed)
        
        e_low = self.energy_bins[:-1]
        e_high = self.energy_bins[1:]
        log_e = torch.log(0.5 * (e_low + e_high))
        log_e_norm = (log_e - log_e.min()) / (log_e.max() - log_e.min())
        log_e_norm = log_e_norm.reshape(self.num_bins, 1) # (N, 1)
        e_encoded = self.energy_encoder(log_e_norm) # (N, energy_enc)
        e_embed = self.energy_embedder(e_encoded) # (N, energy_embed)
        
        # Expand to create all combinations
        _, n_pos = xy_embed.shape
        _, n_energy = e_embed.shape
        xy_expanded = xy_embed.unsqueeze(1).expand(B, self.num_bins, n_pos).reshape(B * self.num_bins, n_pos)
        e_expanded = e_embed.unsqueeze(0).expand(B, self.num_bins, n_energy).reshape(B * self.num_bins, n_energy)

        # Evaluate the hybrid model
        log_f = self.combiner(xy_expanded, e_expanded)
        log_f = log_f.reshape(B, self.num_bins)
        log_f = torch.clamp_max(log_f, self.max_log_flux)
        f = torch.pow(10.0, log_f)

        return {
            "xy_embed": xy_embed,
            "e_embed": e_embed,
            "log_f": log_f,
            "f": f
        }
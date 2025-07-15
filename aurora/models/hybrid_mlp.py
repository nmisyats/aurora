import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class HybridMLP(FluxModel):
    """Hybrid model using two MLPs for position and energy embedding"""
    def __init__(
            self,
            config: ModelConfig,
            position_embed: int = 8,
            energy_embed: int = 8,
            position_enc: int = 4,
            energy_enc: int = 4,
            max_log_flux: float = 7.0,
            embed_hidden_size: int = 64
        ):
        super().__init__(config)

        self.max_log_flux = max_log_flux

        self.position_encoder = ann.FourierEncoder(position_enc)
        self.energy_encoder = ann.FourierEncoder(energy_enc)

        position_encode_dim = self.position_encoder.output_dim(2)
        if position_embed:
            position_embed_dim = position_embed
            self.position_embedder = ann.create_mlp(
                position_encode_dim,
                embed_hidden_size,
                position_embed_dim
            )
        else:
            self.position_embedder = nn.Identity()
            position_embed_dim = position_encode_dim

        energy_encode_dim = self.energy_encoder.output_dim(1)
        if energy_embed:
            energy_embed_dim = energy_embed
            self.energy_embedder = ann.create_mlp(
                energy_encode_dim,
                embed_hidden_size,
                energy_embed_dim
            )
        else:
            self.energy_embedder = nn.Identity()
            energy_embed_dim = energy_encode_dim

        self.combiner = nn.Bilinear(position_embed_dim, energy_embed_dim, 1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.xavier_normal_(self.combiner.weight)
        nn.init.constant_(self.combiner.bias, self.max_log_flux / 2.0)
    
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
        xy_norm = self.bbox.norm_xy(xy) # (B, 2)
        xy_encoded = self.position_encoder(xy_norm) # (B, n_enc)
        xy_embed = self.position_embedder(xy_encoded) # (B, n_embed)
        
        log_e = torch.log(0.5 * (self.energy_bins[:-1] + self.energy_bins[1:])) # (N)
        log_e_norm = (log_e - log_e.min()) / (log_e.max() - log_e.min()) # (N)
        log_e_norm = log_e_norm.unsqueeze(-1) # (N, 1)
        e_encoded = self.energy_encoder(log_e_norm) # (N, m_enc)
        e_embed = self.energy_embedder(e_encoded) # (N, m_embed)

        B, n_embed = xy_embed.shape
        N, m_embed = e_embed.shape
        # (B, n_embed) -> (B, 1, n_embed) -> (B, N, n_embed) -> (B * N, n_embed)
        xy_expanded = xy_embed.unsqueeze(1).expand(B, N, n_embed).reshape(B * N, n_embed)
        # (N, m_embed) -> (1, N, m_embed) -> (B, N, m_embed) -> (B * N, m_embed)
        e_expanded = e_embed.unsqueeze(0).expand(B, N, m_embed).reshape(B * N, m_embed)

        # Evaluate the hybrid model
        combined = self.combiner(xy_expanded, e_expanded) # shape (B * N, 1)
        log_f = F.softplus(combined)
        log_f = log_f.reshape(B, self.num_bins) # shape (B, N)
        f = ann.clamped_exp10(log_f, 0.0, self.max_log_flux)

        return {
            "xy_embed": xy_embed,
            "e_embed": e_embed,
            "combined": combined,
            "log_f": log_f,
            "f": f
        }
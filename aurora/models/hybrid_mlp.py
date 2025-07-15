import torch
import torch.nn as nn

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
            embed_hidden_size: int = 64,
            num_hidden: int = 3,
            hidden_size: int = 128
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

        combined_dim = position_embed_dim + energy_embed_dim
        hidden_sizes = [hidden_size] * num_hidden
        self.combined_mlp = ann.create_mlp(combined_dim, *hidden_sizes, 1)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
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
        
        xy_norm = self.bbox.norm_xy(xy)
        
        # Create all (x,y,E) combinations efficiently
        # if self.training:
        #     # Random sampling within each bin during training
        #     rand_uniform = torch.rand(num_bins, device=xy.device)
        #     # Linear interpolation in log space
        #     log_e_low = torch.log(self.energy_bins[:-1])
        #     log_e_high = torch.log(self.energy_bins[1:])
        #     log_e = log_e_low + rand_uniform * (log_e_high - log_e_low)
        #     e = torch.exp(log_e)
        # else:
        #     # Use geometric mean during inference for consistency
        #     e = torch.sqrt(self.energy_bins[1:] * self.energy_bins[:-1])
        # e_norm = self._normalize_energy(e)

        
        log_e = torch.log(self.energy_bins)
        log_e_norm = (log_e - log_e.min()) / (log_e.max() - log_e.min())
        num_edges = self.num_bins + 1
        
        # Expand to create all combinations: (B*N, 3)
        xy_expanded = xy_norm.unsqueeze(1).expand(B, num_edges, 2).reshape(B * num_edges, 2)
        e_expanded = log_e_norm.unsqueeze(0).expand(B, num_edges).reshape(B * num_edges, 1)

        # Evaluate the hybrid model
        xy_encoded = self.position_encoder(xy_expanded)
        e_encoded = self.energy_encoder(e_expanded)
        xy_embed = self.position_embedder(xy_encoded)
        e_embed = self.energy_embedder(e_encoded)
        combined_input = torch.cat((xy_embed, e_embed), dim=1)
        log_f_at_edges = self.combined_mlp(combined_input)
        f_at_edges = ann.clamped_exp10(log_f_at_edges, 0.0, self.max_log_flux)
        f_at_edges = f_at_edges.reshape(B, num_edges) # Reshape back to (B, N)
        f = 0.5 * (f_at_edges[:, :-1] + f_at_edges[:, 1:])

        return {
            "xy_embed": xy_embed,
            "e_embed": e_embed,
            "xye_embed": combined_input,
            "log_f_at_edges": log_f_at_edges,
            "f_at_edges": f_at_edges,
            "f": f
        }
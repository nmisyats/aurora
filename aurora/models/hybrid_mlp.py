import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class HybridMLP(FluxModel):
    """Hybrid model using two MLPs for position and energy embedding"""
    def __init__(
            self,
            frame: Frame,
            bbox: BBox,
            emis_mat: torch.Tensor,
            dens_mat: torch.Tensor,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            position_embed: int = 8,
            energy_embed: int = 8,
            position_enc: int = 4,
            energy_enc: int = 4,
            max_log_flux: float = 7.0
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

        self.max_log_flux = max_log_flux

        self.position_encoder = ann.FourierEncoder(position_enc)
        self.energy_encoder = ann.FourierEncoder(energy_enc)

        position_encode_dim = self.position_encoder.output_dim(2)
        if position_embed:
            position_embed_dim = position_embed
            self.position_embedder = nn.Sequential(
                nn.Linear(position_encode_dim, 128), nn.ReLU(),
                nn.Linear(128, position_embed_dim)
            )
        else:
            self.position_embedder = nn.Identity()
            position_embed_dim = position_encode_dim

        energy_encode_dim = self.energy_encoder.output_dim(1)
        if energy_embed:
            energy_embed_dim = energy_embed
            self.energy_embedder = nn.Sequential(
                nn.Linear(energy_encode_dim, 128), nn.ReLU(),
                nn.Linear(128, energy_embed_dim)
            )
        else:
            self.energy_embedder = nn.Identity()
            energy_embed_dim = energy_encode_dim

        combined_dim = position_embed_dim + energy_embed_dim
        self.combined_mlp = nn.Sequential(
            nn.Linear(combined_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.exp10 = ann.Exponentiate(10.0, 0.0, self.max_log_flux)

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
        return self._forward_batch(xy.unsqueeze(0)).squeeze(0)

    def _forward_batch(self, xy: torch.Tensor):
        # xy: (B, 2)
        B = xy.shape[0]
        num_bins = len(self.energy_bins) - 1
        
        xy_norm = self._normalize_xy(xy)
        
        # Create all (x,y,E) combinations efficiently
        if self.training:
            # Random sampling within each bin during training
            rand_uniform = torch.rand(num_bins, device=xy.device)
            # Linear interpolation in log space
            log_e_low = torch.log(self.energy_bins[:-1])
            log_e_high = torch.log(self.energy_bins[1:])
            log_e = log_e_low + rand_uniform * (log_e_high - log_e_low)
            e = torch.exp(log_e)
        else:
            # Use geometric mean during inference for consistency
            e = torch.sqrt(self.energy_bins[1:] * self.energy_bins[:-1])
        e_norm = self._normalize_energy(e)
        
        # Expand to create all combinations: (B*N, 3)
        xy_expanded = xy_norm.unsqueeze(1).expand(B, num_bins, 2).reshape(B * num_bins, 2)
        e_expanded = e_norm.unsqueeze(0).expand(B, num_bins).reshape(B * num_bins, 1)

        # Evaluate the hybrid model
        xy_encoded = self.position_encoder(xy_expanded)
        e_encoded = self.energy_encoder(e_expanded)
        xy_embed = self.position_embedder(xy_encoded)
        e_embed = self.energy_embedder(e_encoded)
        combined_input = torch.cat((xy_embed, e_embed), dim=1)
        log_flux = self.combined_mlp(combined_input)
        flux = self.exp10(log_flux)
        
        # Reshape back to (B, N)
        flux = flux.reshape(B, num_bins)
        return {
            "xy_embed": xy_embed,
            "e_embed": e_embed,
            "comb_embed": combined_input,
            "log_f": log_flux,
            "f": flux
        }
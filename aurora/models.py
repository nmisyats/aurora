from collections import namedtuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import aurora.dnn as ann


ModelEntry = namedtuple("ModelEntry", ("model_cls", "config_cls"))


def config_field(default=None, *, help: str | None = None):
    kwargs = {}
    if default is not None:
        kwargs["default"] = default
    if help is not None:
        kwargs["metadata"] = {"help": help}
    return field(**kwargs)

MODEL_REGISTRY: dict[str, ModelEntry] = {}

def register_model(name, config_cls):
    def decorator(model_cls):
        MODEL_REGISTRY[name] = ModelEntry(model_cls, config_cls)
        return model_cls
    return decorator



class FluxModel(nn.Module, ABC):
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
        ):
        super().__init__()

        self.register_buffer("xy_min", xy_min)
        self.register_buffer("xy_max", xy_max)
        self.register_buffer("energy_bins", energy_bins)

        self.num_bins = len(energy_bins) - 1
    
    @abstractmethod
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (n, 2)
        ...
    
    def normalize_energy(self, E: torch.Tensor):
        E_min = self.energy_bins[0]
        E_max = self.energy_bins[-1]
        return (E - E_min) / (E_max - E_min)
    
    def normalize_xy(self, xy: torch.Tensor):
        xy_min, xy_max = self.xy_min, self.xy_max
        return (xy - xy_min) / (xy_max - xy_min)


class GridSampledFlux(FluxModel):
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            data: torch.Tensor
        ):
        super().__init__(xy_min, xy_max, energy_bins)
        
        self.register_buffer("data", data)
    
    @property
    def resolution(self):
        res_y, res_x, _ = self.data.shape
        return res_x, res_y
    
    def forward(self, xy: torch.Tensor):
        # xy: (N, 2) coordinates within xy_min and xy_max
        # self.data: (H, W, B)
        # Output: (N, B) sampled flux at each xy

        H, W, B = self.data.shape

        # Normalize xy to [0, 1]
        norm_xy = self.normalize_xy(xy)
        norm_xy = torch.clamp(norm_xy, 0, 1)

        # Scale to image pixel coordinates
        y_idx = norm_xy[:, 1] * (H - 1)
        x_idx = norm_xy[:, 0] * (W - 1)

        # Create grid for grid_sample
        grid = torch.stack((x_idx, y_idx), dim=1).unsqueeze(0).unsqueeze(2)  # (1, N, 1, 2)
        grid = 2 * grid / torch.tensor([W - 1, H - 1], device=xy.device) - 1  # Normalize to [-1, 1]
        grid = grid[..., [1, 0]]  # switch x, y -> y, x
        grid = grid.expand(B, -1, -1, -1)  # (B, N, 1, 2)

        # Prepare input image tensor for grid_sample
        flux = self.data.permute(2, 0, 1).unsqueeze(1)  # (B, 1, H, W)

        # Perform bilinear sampling
        sampled = torch.nn.functional.grid_sample(
            flux, grid, mode='bilinear', align_corners=True
        )  # (B, 1, N, 1)

        # Reshape result to (N, B)
        output = sampled.squeeze(3).squeeze(1).T  # (N, B)
        return output


@dataclass
class MLPConfig:
    encoding_exp: int = config_field(4, help="Fourier embedding maximum exponent")
    max_log_flux: float = config_field(7.0, help="Maximum logarithmic value of the flux")

@register_model("spectral_mlp", MLPConfig)
class SpectralMLP(FluxModel):
    """MLP outputing the energy spectrum from the xy position"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.max_log_flux = config.max_log_flux

        encode_dim = self.fourier_encoder.output_dim(2)

        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.num_bins)
        )

        self.exp10 = ann.Exponentiate(10.0, 0.0, self.max_log_flux)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
    def forward(self, xy: torch.Tensor):
        x = self.normalize_xy(xy)
        x = self.fourier_encoder(x)
        x = self.mlp(x)
        x = self.exp10(x)
        return x

@register_model("spectral_res_mlp", MLPConfig)
class SpectralResMLP(FluxModel):
    """Spectral MLP with a residual connection"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.max_log_flux = config.max_log_flux

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
        xy = self.normalize_xy(xy)
        x = self.fourier_encoder(xy)
        x0 = x
        x = self.mlp1(x)
        x = torch.cat([x, x0], dim=-1)
        x = self.mlp2(x)
        x = self.exp10(x)
        return x

@register_model("direct_mlp", MLPConfig)
class DirectMLP(FluxModel):
    """MLP with position and energy input"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.max_log_flux = config.max_log_flux

        encode_dim = self.fourier_encoder.output_dim(3)

        self.mlp = nn.Sequential(
            nn.Linear(encode_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.exp10 = ann.Exponentiate(10.0, 0.0, self.max_log_flux)

        nn.init.normal_(self.mlp[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.mlp[-1].bias, self.max_log_flux / 2.0)
    
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
        xy_norm = self.normalize_xy(xy)
        
        # Create all (x,y,E) combinations efficiently
        if self.training:
            # Random sampling within each bin during training
            rand_uniform = torch.rand(self.num_bins, device=xy.device)
            # Linear interpolation in log space
            log_e_low = torch.log(self.energy_bins[:-1])
            log_e_high = torch.log(self.energy_bins[1:])
            log_e = log_e_low + rand_uniform * (log_e_high - log_e_low)
            e = torch.exp(log_e)
        else:
            # Use geometric mean during inference for consistency
            e = torch.sqrt(self.energy_bins[1:] * self.energy_bins[:-1])
        e_norm = self.normalize_energy(e)
        
        # Expand to create all combinations: (B*N, 3)
        xy_expanded = xy_norm.unsqueeze(1).expand(B, self.num_bins, 2).reshape(B * self.num_bins, 2)
        e_expanded = e_norm.unsqueeze(0).expand(B, self.num_bins).reshape(B * self.num_bins, 1)
        x_input = torch.cat((xy_expanded, e_expanded), dim=1)  # (B*N, 3)
        
        # Single forward pass through the network
        x = self.fourier_encoder(x_input)  # (B*N, encoding_size)
        x = self.mlp(x)
        x = self.exp10(x)
        
        # Reshape back to (B, N)
        return x.reshape(B, self.num_bins)


@dataclass
class BasisMLPConfig(MLPConfig):
    num_basis: int = config_field(8, help="Number of basis parameters")

@register_model("poly_mlp", BasisMLPConfig)
class PolyMLP(FluxModel):
    """MLP learning a polynomial basis of the flux"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            config: BasisMLPConfig = BasisMLPConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)

        self.num_basis = config.num_basis
        self.max_log_flux = config.max_log_flux

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        encoding_size = self.fourier_encoder.output_dim(2)
        
        self.mlp = nn.Sequential(
            nn.Linear(encoding_size, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.num_basis)
        )
        
        # Pre-compute basis functions
        self.register_buffer('basis_functions', self._create_basis_functions())
    
    def _create_basis_functions(self):
        # Work in normalized log-energy space
        e_norm = self.normalize_energy(self.energy_bins) + 1e-8
        log_energies = torch.log(e_norm)
        log_e_min = log_energies.min()
        log_e_max = log_energies.max()
        log_e_norm = (log_energies - log_e_min) / (log_e_max - log_e_min)
        # Create basis functions
        basis = torch.zeros(self.num_basis, self.num_bins + 1)
        for i in range(self.num_basis):
            basis[i] = log_e_norm ** i
        return basis # (num_basis, num_edges)
    
    def forward(self, xy: torch.Tensor):
        xy = self.normalize_xy(xy)
        x = self.fourier_encoder(xy)
        coeffs = self.mlp(x) # (batch_size, num_basis + 1)
        log_f_edges = torch.matmul(coeffs, self.basis_functions) # (batch_size, num_edges)
        log_f = 0.5 * (log_f_edges[:, :-1] + log_f_edges[:, 1:]) # (batch_size, num_bins)
        f = torch.exp(log_f)
        f = torch.clamp(f, min=1.0, max=10.0**self.max_log_flux)
        return f


@dataclass
class HybridMLPConfig:
    position_embed: int = config_field(0, help="Size of the position embedding (use 0 for no embedding)")
    energy_embed: int = config_field(8, help="Size of the energy embedding (use 0 for no embedding)")
    position_exp: int = config_field(4, help="Maximum exponent for position fourier encoding")
    energy_exp: int = config_field(4, help="Maximum exponent for energy fourier encoding")
    max_log_flux: float = config_field(7.0, help="Maximum logarithmic value of the flux")

@register_model("hybrid_mlp", HybridMLPConfig)
class HyrbidMLP(FluxModel):
    """Hybrid model using two MLPs for position and energy embedding"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            energy_bins: torch.Tensor,
            config: HybridMLPConfig = HybridMLPConfig()
        ):
        super().__init__(xy_min, xy_max, energy_bins)

        self.max_log_flux = config.max_log_flux

        self.position_encoder = ann.FourierEncoder(config.position_exp)
        self.energy_encoder = ann.FourierEncoder(config.energy_exp)

        position_encode_dim = self.position_encoder.output_dim(2)
        if config.position_embed:
            position_embed_dim = config.position_embed
            self.position_embedder = nn.Sequential(
                nn.Linear(position_encode_dim, 128), nn.ReLU(),
                nn.Linear(128, position_embed_dim)
            )
        else:
            self.position_embedder = nn.Identity()
            position_embed_dim = position_encode_dim

        energy_encode_dim = self.energy_encoder.output_dim(1)
        if config.energy_embed:
            energy_embed_dim = config.energy_embed
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
        xy_norm = self.normalize_xy(xy)
        
        # Create all (x,y,E) combinations efficiently
        if self.training:
            # Random sampling within each bin during training
            rand_uniform = torch.rand(self.num_bins, device=xy.device)
            # Linear interpolation in log space
            log_e_low = torch.log(self.energy_bins[:-1])
            log_e_high = torch.log(self.energy_bins[1:])
            log_e = log_e_low + rand_uniform * (log_e_high - log_e_low)
            e = torch.exp(log_e)
        else:
            # Use geometric mean during inference for consistency
            e = torch.sqrt(self.energy_bins[1:] * self.energy_bins[:-1])
        e_norm = self.normalize_energy(e)
        
        # Expand to create all combinations: (B*N, 3)
        xy_expanded = xy_norm.unsqueeze(1).expand(B, self.num_bins, 2).reshape(B * self.num_bins, 2)
        e_expanded = e_norm.unsqueeze(0).expand(B, self.num_bins).reshape(B * self.num_bins, 1)

        # Evaluate the hybrid model
        xy_encoded = self.position_encoder(xy_expanded)
        e_encoded = self.energy_encoder(e_expanded)
        xy_embed = self.position_embedder(xy_encoded)
        e_embed = self.energy_embedder(e_encoded)
        x_input = torch.cat((xy_embed, e_embed), dim=1)
        log_flux = self.combined_mlp(x_input)
        flux = self.exp10(log_flux)
        
        # Reshape back to (B, N)
        return flux.reshape(B, self.num_bins)
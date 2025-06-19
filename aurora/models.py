from collections import namedtuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import aurora.dnn as ann


ModelEntry = namedtuple("RegisteredModel", ("model_cls", "config_cls"))


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
            E_edges: torch.Tensor,
        ):
        super().__init__()

        self.register_buffer("xy_min", xy_min)
        self.register_buffer("xy_max", xy_max)
        self.register_buffer("E_edges", E_edges)

        self.input_dim = 2
        self.output_dim = len(E_edges) - 1
    
    @abstractmethod
    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (n, 2)
        ...
    
    def normalize_energy(self, E: torch.Tensor):
        E_min, E_max = self.E_edges[0], self.E_edges[-1]
        return (E - E_min) / (E_max - E_min)
    
    def normalize_xy(self, xy: torch.Tensor):
        xy_min, xy_max = self.xy_min, self.xy_max
        return (xy - xy_min) / (xy_max - xy_min)


class StaticFlux(FluxModel):
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            E_edges: torch.Tensor,
            data: torch.Tensor
        ):
        super().__init__(xy_min, xy_max, E_edges)
        
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
    log_scale: float = config_field(7.0, help="Logarithmic range of the flux")

@register_model("spectral_mlp", MLPConfig)
class SpectralMLP(FluxModel):
    """MLP outputing the energy spectrum from the xy position"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            E_edges: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, E_edges)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.log_scale = config.log_scale

        self.mlp = nn.Sequential(
            nn.Linear(self.fourier_encoder.output_dim(2), 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, self.output_dim), nn.Sigmoid()
        )
    
    def forward(self, xy: torch.Tensor):
        x = self.normalize_xy(xy)
        x = self.fourier_encoder(x)
        x = self.mlp(x)
        x = torch.pow(10.0, x * self.log_scale)
        return x

@register_model("spectral_res_mlp", MLPConfig)
class SpectralResMLP(FluxModel):
    """Spectral MLP with a residual connection"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            E_edges: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, E_edges)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.log_scale = config.log_scale

        encode_dim = self.fourier_encoder.output_dim(2)

        self.mlp1 = nn.Sequential(
            nn.Linear(encode_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(128 + encode_dim, 128), nn.ReLU(),
            nn.Linear(128, self.output_dim), nn.Sigmoid()
        )
    
    def forward(self, xy: torch.Tensor):
        xy = self.normalize_xy(xy)
        x = self.fourier_encoder(xy)
        x0 = x
        x = self.mlp1(x)
        x = torch.cat([x, x0], dim=-1)
        x = self.mlp2(x)
        x = torch.pow(10.0, x * self.log_scale)
        return x

@register_model("direct_mlp", MLPConfig)
class DirectMLP(FluxModel):
    """MLP with position and energy input"""
    def __init__(
            self,
            xy_min: torch.Tensor,
            xy_max: torch.Tensor,
            E_edges: torch.Tensor,
            config: MLPConfig = MLPConfig()
        ):
        super().__init__(xy_min, xy_max, E_edges)

        self.fourier_encoder = ann.FourierEncoder(config.encoding_exp)
        self.log_scale = config.log_scale

        self.mlp = nn.Sequential(
            nn.Linear(self.fourier_encoder.output_dim(3), 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    
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
            rand_uniform = torch.rand(self.output_dim, device=xy.device)
            # Linear interpolation in log space
            log_e_low = torch.log(self.E_edges[:-1])
            log_e_high = torch.log(self.E_edges[1:])
            log_e = log_e_low + rand_uniform * (log_e_high - log_e_low)
            e = torch.exp(log_e)
        else:
            # Use geometric mean during inference for consistency
            e = torch.sqrt(self.E_edges[1:] * self.E_edges[:-1])
        e_norm = self.normalize_energy(e)
        
        # Expand to create all combinations: (B*N, 3)
        xy_expanded = xy_norm.unsqueeze(1).expand(B, self.output_dim, 2).reshape(B * self.output_dim, 2)
        e_expanded = e_norm.unsqueeze(0).expand(B, self.output_dim).reshape(B * self.output_dim, 1)
        x_input = torch.cat((xy_expanded, e_expanded), dim=1)  # (B*N, 3)
        
        # Single forward pass through the network
        x = self.fourier_encoder(x_input)  # (B*N, encoding_size)
        x = self.mlp(x)
        x = torch.pow(10.0, x * self.log_scale)
        
        # Reshape back to (B, N)
        return x.reshape(B, self.output_dim)


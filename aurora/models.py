from collections import namedtuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.bbox import BBox


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



class FluxModel(ABC):
    @abstractmethod
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        ...
    
    @property
    @abstractmethod
    def E_edges(self) -> torch.Tensor:
        ...
    
    @property
    @abstractmethod
    def xy_min(self) -> torch.Tensor:
        ...
    
    @property
    @abstractmethod
    def xy_max(self) -> torch.Tensor:
        ...


class TrainableFluxModel(FluxModel, nn.Module):
    def flux(self, xy: torch.Tensor):
        orig_shape = xy.shape[:-1] # (k1, k2, ..., kn, 2)
        xy = xy.reshape(-1, 2) # (N, 2)
        f = self.forward(xy) # (N, n_bins)
        n_bins = f.shape[-1]
        return f.reshape(*orig_shape, n_bins)

    @classmethod
    def default_config(cls):
        return None


class ReferenceFlux(FluxModel):
    def __init__(
            self,
            image: torch.Tensor,
            E_edges: torch.Tensor,
            range_south: tuple[float, float],
            range_east: tuple[float, float],
            device: torch.device
        ):
        self.image = image.to(device)
        self.device = device
        x_min, x_max = range_south
        y_min, y_max = range_east
        self._xy_min = torch.tensor([x_min, y_min], device=device)
        self._xy_max = torch.tensor([x_max, y_max], device=device)
        self._E_edges = E_edges.to(device)
    
    @property
    def xy_min(self):
        return self._xy_min
    
    @property
    def xy_max(self):
        return self._xy_max
    
    @property
    def E_edges(self):
        return self._E_edges
    
    @property
    def resolution(self):
        res_x = self.image.size(1)
        res_y = self.image.size(0)
        return res_x, res_y
    
    def flux(self, xy: torch.Tensor) -> torch.Tensor:
        # xy: (..., 2) coordinates within xy_min and xy_max
        # self.image: (H, W, B)
        # Output: (..., B) sampled flux at each xy

        H, W, B = self.image.shape

        # Save original shape (excluding the last dimension, which is 2)
        orig_shape = xy.shape[:-1]  # (k1, ..., kn)
        xy_flat = xy.reshape(-1, 2)  # (N, 2)

        # Normalize xy to [0, 1]
        norm_xy = (xy_flat - self.xy_min) / (self.xy_max - self.xy_min)
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
        flux = self.image.permute(2, 0, 1).unsqueeze(1)  # (B, 1, H, W)

        # Perform bilinear sampling
        sampled = torch.nn.functional.grid_sample(
            flux, grid, mode='bilinear', align_corners=True
        )  # (B, 1, N, 1)

        # Reshape result to (..., B)
        output = sampled.squeeze(3).squeeze(1).T  # (N, B)
        return output.reshape(*orig_shape, B)     # (..., B)


class NNFluxModel(TrainableFluxModel):
    def __init__(self, bbox: BBox, E_edges: torch.Tensor):
        super(NNFluxModel, self).__init__()
        self._xy_min = bbox.xyz_min[:2]
        self._xy_max = bbox.xyz_max[:2]
        self._E_edges = E_edges
        self.input_size = 2
        self.output_size = len(E_edges) - 1  # Number of energy bins
    
    def normalize_xy(self, xy: torch.Tensor):
        return (xy - self._xy_min) / (self._xy_max - self._xy_min)
    
    @property
    def xy_min(self):
        return self._xy_min
    
    @property
    def xy_max(self):
        return self._xy_max
    
    @property
    def E_edges(self):
        return self._E_edges


class FourierNNFluxModel(NNFluxModel):
    def __init__(self, bbox: BBox, E_edges: torch.Tensor, embed_exp: int):
        assert embed_exp >= 1

        super().__init__(bbox, E_edges)
        
        self.embed_exp = embed_exp
        self.embed_size = (2 * embed_exp + 1) * self.input_size
        self.freqs = [(2**i) * torch.pi for i in range(self.embed_exp)]
    
    def position_encode(self, x: torch.Tensor):
        cos_x = [torch.cos(f * x) for f in self.freqs]
        sin_x = [torch.sin(f * x) for f in self.freqs]
        return torch.cat((x, *cos_x, *sin_x), dim=-1)


@dataclass
class LogMLPConfig:
    embed_exp: int = config_field(4, help="Fourier embedding maximum exponent")
    log_scale: float = config_field(7.0, help="Logarithmic range of the flux")

@register_model("log_res_mlp", LogMLPConfig)
class LogResMLP(FourierNNFluxModel):
    def __init__(self, bbox: BBox, E_edges: torch.Tensor, config: LogMLPConfig):
        assert config.embed_exp >= 1

        super().__init__(bbox, E_edges, config.embed_exp)

        self.log_scale = config.log_scale
        
        self.fc1 = nn.Linear(self.embed_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + self.embed_size, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.output_size)
    
    def forward(self, xy: torch.Tensor):
        xy = self.normalize_xy(xy)
        x = self.position_encode(xy)
        x0 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, x0], dim=-1)))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = x * self.log_scale
        x = torch.pow(10.0, x)
        return x

    @classmethod
    def default_config(cls):
        return LogMLPConfig()
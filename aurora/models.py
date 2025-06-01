from collections import namedtuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from aurora.physics import PhysicalModel, ElectronFluxModel, TrainableFluxModel


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



class ReferenceFlux(ElectronFluxModel):
    def __init__(self, image: torch.Tensor, oblique_range_x: tuple[float, float], oblique_range_y: tuple[float, float], device: torch.device):
        self.image = image.to(device)
        self.device = device
        x_min, x_max = oblique_range_x
        y_min, y_max = oblique_range_y
        self.xy_min = torch.tensor([x_min, y_min], device=device)
        self.xy_max = torch.tensor([x_max, y_max], device=device)
    
    @property
    def resolution(self):
        res_x = self.image.size(1)
        res_y = self.image.size(0)
        return res_x, res_y
    
    def f_at(self, xy: torch.Tensor) -> torch.Tensor:
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


@dataclass
class LogMLPConfig:
    embed_exp: int = config_field(4, help="Fourier embedding maximum exponent")
    log_scale: float = config_field(7.0, help="Logarithmic range of the flux")

@register_model("log_res_mlp", LogMLPConfig)
class LogResMLP(TrainableFluxModel):
    def __init__(self, pm: PhysicalModel, config: LogMLPConfig):
        assert config.embed_exp >= 1

        super(LogResMLP, self).__init__()

        self.xy_min = pm.frame.xy_min
        self.xy_max = pm.frame.xy_max
        self.input_size = 2
        self.embed_exp = config.embed_exp
        self.embed_size = self.input_size + self.input_size * 2 * config.embed_exp
        self.output_size = len(pm.E_edges) - 1
        self.log_scale = config.log_scale
        
        self.fc1 = nn.Linear(self.embed_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128 + self.embed_size, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.output_size)
    
    def forward(self, xy: torch.Tensor):
        xy = (xy - self.xy_min) / (self.xy_max - self.xy_min)
        x = self.embed_fourier(xy)
        x0 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(torch.cat([x, x0], dim=-1)))
        x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        x = x * self.log_scale
        x = torch.pow(10.0, x)
        return x
    
    def embed_fourier(self, x: torch.Tensor):
        freqs = [(2**i) * torch.pi for i in range(self.embed_exp)]
        cos_x = [torch.cos(f * x) for f in freqs]
        sin_x = [torch.sin(f * x) for f in freqs]
        return torch.cat((x, *cos_x, *sin_x), dim=-1)

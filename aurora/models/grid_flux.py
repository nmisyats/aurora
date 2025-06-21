import torch

from aurora.models.flux_model import FluxModel
from aurora.frame import Frame
from aurora.bbox import BBox


class GridSampledFlux(FluxModel):
    def __init__(
            self,
            frame: Frame,
            bbox: BBox,
            emis_mat: torch.Tensor,
            dens_mat: torch.Tensor,
            altitude_bins: torch.Tensor,
            energy_bins: torch.Tensor,
            data: torch.Tensor
        ):
        super().__init__(frame, bbox, emis_mat, dens_mat, altitude_bins, energy_bins)

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
        norm_xy = self._normalize_xy(xy)
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
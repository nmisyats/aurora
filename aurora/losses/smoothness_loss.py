import torch

from aurora.losses.loss_term import LossTerm
from aurora.models.flux_model import FluxModel


class SpectralSmoothnessLoss(LossTerm):
    def __init__(
            self,
            model: FluxModel,
            weight: float = 0.01,
            batch_size: int = 1024,
            name: str = "flux_smooth"
        ):
        super().__init__(model, weight, batch_size, name)
    
    def sample_batch(self) -> torch.Tensor:
        xy_min, xy_max = self.model.bbox.xy_min, self.model.bbox.xy_max
        device = self.model.device
        rand = torch.rand(self.batch_size, 2, device=device)
        xy_samples = xy_min + rand * (xy_max - xy_min)
        return xy_samples
    
    def eval_raw_loss(self, xy_samples: torch.Tensor) -> torch.Tensor:
        f_pred = self.model.flux(xy_samples)
        # Calculate bin centers and widths
        energy_bins = self.model.energy_bins
        bin_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])  # (num_bins,)
        # bin_widths = energy_bins[1:] - energy_bins[:-1]  # (num_bins,)
        # Log-scale bin centers for proper weighting
        log_centers = torch.log(bin_centers)
        log_spacing = torch.diff(log_centers)  # (num_bins-1,)
        # Second-order derivative approximation in log-energy space
        # d²f/d(log E)² ≈ [f(i+1) - f(i)]/Δ(log E) - [f(i) - f(i-1)]/Δ(log E)
        first_diff = torch.diff(f_pred, dim=1)  # (batch_size, num_bins-1)
        # Normalize by log spacing
        normalized_diff = first_diff / log_spacing.unsqueeze(0)  # (batch_size, num_bins-1)
        # Second derivative
        second_diff = torch.diff(normalized_diff, dim=1)  # (batch_size, num_bins-2)
        # L2 smoothness penalty
        smoothness_penalty = torch.mean(second_diff**2)
        return smoothness_penalty
from abc import ABC, abstractmethod
from typing import Callable
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import tqdm
from aurora.reconstruction import Reconstruction
from aurora.dataset import CameraRaysDataset, RadarPointsDataset

# Type hints for clarity
LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
RegularizationFunction = Callable[[torch.Tensor], torch.Tensor]

# Simple configuration for each loss term
class LossTerm(ABC):
    """Configuration for a single loss term"""
    def __init__(self, 
                 weight: float = 1.0,
                 batch_size: int = 1000,
                 name: str = None):
        self.weight = weight
        self.batch_size = batch_size
        self.name = name if name is not None else self.__class__.__name__.lower()
        self.history = []
    
    def eval_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        """Compute weighted loss for this term"""
        loss = self.eval_raw_batch_loss(recon)
        weighted_loss = self.weight * loss
        self.history.append(weighted_loss.item())
        return weighted_loss
    
    @abstractmethod
    def eval_raw_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        """Compute loss for this term"""
        ...

# Convenience classes for common use cases
class RayLoss(LossTerm):
    def __init__(
            self,
            ray_data: CameraRaysDataset, 
            batch_size: int = 4096,
            weight: float = 1.0, 
            ray_bins: int = 100,
            loss_fn: LossFunction = nn.MSELoss(),
            name: str = "ray_loss"
        ):
        super().__init__(weight, batch_size, name)
        self.ray_data = ray_data
        self.ray_bins = ray_bins
        self.loss_fn = loss_fn
    
    def eval_raw_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        ro, rd, tn, tf, g_target = self.ray_data.random_sample(self.batch_size)
        g_pred = recon.int_emis_ray(ro, rd, tn, tf, self.ray_bins)
        return self.loss_fn(g_pred, g_target)

class RadarLoss(LossTerm):
    def __init__(
            self,
            radar_data: RadarPointsDataset,
            batch_size: int = 1024,
            weight: float = 1.0,
            loss_fn: LossFunction = nn.MSELoss(),
            name: str = "radar_loss"
        ):
        super().__init__(weight, batch_size, name)
        self.radar_data = radar_data
        self.loss_fn = loss_fn
    
    def eval_raw_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        p, d_target = self.radar_data.random_sample(self.batch_size)
        d_pred = recon.elec_dens(p)
        return self.loss_fn(d_pred, d_target)

class SpectralSmoothnessLoss(LossTerm):
    def __init__(
            self,
            batch_size: int = 1024,
            weight: float = 0.01,
            name: str = "flux_smooth"
        ):
        super().__init__(weight, batch_size, name)
    
    def eval_raw_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        xy_min, xy_max = recon.bbox.xy_min, recon.bbox.xy_max
        device = recon.device
        xy_samples = torch.rand(self.batch_size, 2, device=device) * (xy_max - xy_min) + xy_min
        f_pred = recon.flux(xy_samples)
        energy_bins = recon.flux_model.energy_bins
        # Calculate bin centers and widths
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

def train(
        recon: Reconstruction,
        loss_terms: LossTerm | list[LossTerm],
        iters: int = 1000,
        lr: float = 5e-5,
        weight_decay: float = 1.0,
        lr_step: int = 1000,
        lr_decay: float = 0.5,
        progress_bar: bool = True
    ) -> dict[str, list[float]]:
    """
    Train reconstruction with flexible loss terms.
    
    Args:
        recon: Reconstruction object to train
        loss_terms: List of LossTerm objects defining the loss function
        ... (other training hyperparameters)
    
    Returns:
        Dictionary mapping loss term names to their training history
    """
    if not loss_terms:
        raise ValueError("At least one loss term must be provided")
    
    if not isinstance(loss_terms, list):
        loss_terms = [loss_terms]
    
    recon.train()
    
    optimizer = torch.optim.Adam(recon.flux_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
    
    total_history = []
    iterator = tqdm.trange(iters) if progress_bar else range(iters)
    
    for iter in iterator:
        optimizer.zero_grad()
        
        total_loss = torch.tensor(0.0, device=recon.device)
        
        # Compute all loss terms
        for term in loss_terms:
            if term.weight > 0:  # Skip disabled terms
                term_loss = term.eval_batch_loss(recon)
                total_loss = total_loss + term_loss
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_history.append(total_loss.item())
        
        if progress_bar:
            # Show recent losses in progress bar
            recent_losses = {term.name: term.history[-1] for term in loss_terms if term.history}
            loss_str = " ".join([f"{name}:{val:.2f}" for name, val in recent_losses.items()])
            iterator.set_postfix_str(f"total:{total_loss.item():.2f} {loss_str} lr:{scheduler.get_last_lr()[0]:.2e}")
    
    # Return all histories
    history = {"total": total_history}
    for term in loss_terms:
        history[term.name] = term.history
    
    return history

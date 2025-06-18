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

# Custom regularization functions (since PyTorch doesn't have these built-in)
def l2_regularization(pred: torch.Tensor) -> torch.Tensor:
    """L2 regularization on predictions"""
    return torch.mean(pred ** 2)

def l1_regularization(pred: torch.Tensor) -> torch.Tensor:
    """L1 regularization on predictions"""
    return torch.mean(torch.abs(pred))

def smoothness_regularization(pred: torch.Tensor) -> torch.Tensor:
    """Spatial smoothness regularization"""
    if pred.ndim < 2 or pred.shape[0] < 2:
        # return torch.tensor(0.0, device=pred.device)
        raise ValueError(f"Invalid shape {pred.shape} for smoothness regularization")
    # Compute differences between adjacent samples
    diff = pred[1:] - pred[:-1]
    return torch.mean(diff ** 2)

def positivity_constraint(pred: torch.Tensor) -> torch.Tensor:
    """Enforce non-negative values"""
    return torch.mean(torch.relu(-pred))

# Data samplers - handle the messy data sampling logic
class DataSampler(ABC):
    """Handles data sampling and prediction generation"""
    
    @abstractmethod
    def sample_batch(self, recon: Reconstruction, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predictions, targets)"""
        ...

class RayDataSampler(DataSampler):
    def __init__(self, ray_data: CameraRaysDataset, ray_bins: int = 100):
        self.ray_data = ray_data
        self.ray_bins = ray_bins
    
    def sample_batch(self, recon: Reconstruction, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        ro, rd, tn, tf, g_target = self.ray_data.random_sample(batch_size)
        g_pred = recon.int_emis_ray(ro, rd, tn, tf, self.ray_bins)
        return g_pred, g_target

class RadarDataSampler(DataSampler):
    def __init__(self, radar_data: RadarPointsDataset):
        self.radar_data = radar_data
    
    def sample_batch(self, recon: Reconstruction, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        p, d_target = self.radar_data.random_sample(batch_size)
        d_pred = recon.elec_dens(p)
        return d_pred, d_target

class RegularizationSampler(DataSampler):
    """Sampler for regularization terms that don't have targets"""
    def __init__(self, sample_fn: Callable[[Reconstruction, int], torch.Tensor]):
        self.sample_fn = sample_fn
    
    def sample_batch(self, recon: Reconstruction, batch_size: int) -> tuple[torch.Tensor, None]:
        pred = self.sample_fn(recon, batch_size)
        return pred, None

# Simple configuration for each loss term
class LossTerm:
    """Configuration for a single loss term"""
    def __init__(self, 
                 loss_fn: LossFunction | RegularizationFunction,
                 sampler: DataSampler,
                 weight: float = 1.0,
                 batch_size: int = 1000,
                 name: str = None):
        self.loss_fn = loss_fn
        self.sampler = sampler
        self.weight = weight
        self.batch_size = batch_size
        self.name = name or getattr(loss_fn, '__name__', loss_fn.__class__.__name__)
        self.history = []
    
    def eval_batch_loss(self, recon: Reconstruction) -> torch.Tensor:
        """Compute weighted loss for this term"""
        pred, target = self.sampler.sample_batch(recon, self.batch_size)
        
        if target is not None:
            # Supervised loss
            loss = self.loss_fn(pred, target)
        else:
            # Regularization loss
            loss = self.loss_fn(pred)
        
        weighted_loss = self.weight * loss
        self.history.append(loss.item())
        return weighted_loss

# Convenience classes for common use cases
class RayLoss(LossTerm):
    def __init__(
            self,
            ray_data: CameraRaysDataset, 
            batch_size: int = 4096,
            weight: float = 1.0, 
            ray_bins: int = 100,
            loss_fn: LossFunction = None,
            name: str = "ray_loss"
        ):
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        sampler = RayDataSampler(ray_data, ray_bins)
        
        super().__init__(loss_fn, sampler, weight, batch_size, name)

class RadarLoss(LossTerm):
    def __init__(
            self,
            radar_data: RadarPointsDataset,
            batch_size: int = 1000,
            weight: float = 1.0,
            loss_fn: LossFunction = None,
            name: str = "radar_loss"
        ):
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        
        sampler = RadarDataSampler(radar_data)
        
        super().__init__(loss_fn, sampler, weight, batch_size, name)

class FluxRegularization(LossTerm):
    def __init__(
            self,
            reg_type: str = "l2",
            batch_size: int = 500,
            weight: float = 0.01,
            name: str = "flux_reg"
        ):
        # Create sampling function
        def sample_flux(recon: Reconstruction, batch_size: int) -> torch.Tensor:
            # Sample random points in the reconstruction domain
            xy_min, xy_max = recon.bbox.xy_min, recon.bbox.xy_max
            device = recon.device
            xy_samples = torch.rand(batch_size, 2, device=device) * (xy_max - xy_min) + xy_min
            return recon.flux(xy_samples)
        
        # Choose regularization loss
        if reg_type == "l2":
            reg_loss = l2_regularization
        elif reg_type == "l1":
            reg_loss = l1_regularization
        elif reg_type == "smoothness":
            reg_loss = smoothness_regularization
        elif reg_type == "positivity":
            reg_loss = positivity_constraint
        else:
            raise ValueError(f"Unknown regularization type: {reg_type}")
        
        sampler = RegularizationSampler(sample_flux)

        super().__init__(reg_loss, sampler, weight, batch_size, name)

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
    if not recon.trainable:
        raise ValueError("Reconstruction is not trainable")
    
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

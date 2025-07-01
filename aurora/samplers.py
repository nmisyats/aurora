from abc import ABC, abstractmethod

import torch

import aurora.geometry as gmt


def sample_equal(
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        num_samples: int
    ):
    """
    Samples equidistant `num_samples` samples between [t_min, t_max], edges included.
    
    Args:
        t_min: (n,) tensor representing starting points.
        t_max: (n,) tensor representing end points.
        num_samples: Number of samples of [min_t, max_t].

    Returns:
        (n, num_samples) tensor.
    """
    device = t_min.device
    n = t_min.shape[0]
    
    t_norm = torch.linspace(0.0, 1.0, num_samples, device=device) # (num_samples,)
    t_norm = t_norm.expand(n, -1) # (n, num_samples)
    
    t_min = t_min.unsqueeze(-1) # (n, 1)
    t_max = t_max.unsqueeze(-1) # (n, 1)
    t = t_min + t_norm * (t_max - t_min) # (n, num_samples)
    return t

def sample_stratified(
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        num_bins: int
    ):
    """
    Splits [t_min, t_max] into `num_bins` uniform bins and samples randomly in
    each bin.
    
    Args:
        t_min: (n,) tensor representing starting points.
        t_max: (n,) tensor representing end points.
        num_bins: Number of samples of [t_min, t_max].

    Returns:
        (n, num_samples) tensor.
    """
    device = t_min.device
    n = t_min.shape[0]
    
    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device) # num_bins + 1 edges (num_bins + 1,)
    bin_edges = bin_edges.expand(n, -1) # (n, num_bins + 1)
    t_min = t_min.unsqueeze(-1) # (n, 1)
    t_max = t_max.unsqueeze(-1) # (n, 1)
    bin_edges = bin_edges * (t_max - t_min) + t_min
    
    # Lower and upper edges of each bin
    lower_edges = bin_edges[:, :-1] # (n, num_bins)
    upper_edges = bin_edges[:, 1:] # (n, num_bins)

    # Generate random values in each bin
    t_norm = torch.rand(n, num_bins, device=device) # (n, num_bins)
    t = lower_edges + t_norm * (upper_edges - lower_edges) # (n, num_bins)
    return t

class RaySampler(ABC):
    @abstractmethod
    def sample_distances(
            self,
            tn: torch.Tensor,
            tf: torch.Tensor
        ) -> torch.Tensor:
        ...
    
    def __call__(
            self,
            ro: torch.Tensor,
            rd: torch.Tensor,
            tn: torch.Tensor,
            tf: torch.Tensor
        ):
        t = self.sample_distances(tn, tf)
        p = gmt.get_ray_points(ro, rd, t)
        return p, t

class EqualSampler(RaySampler):
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
    
    def sample_distances(self, tn, tf):
        return sample_equal(tn, tf, self.num_samples)

class StratifiedSampler(RaySampler):
    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
    
    def sample_distances(self, tn, tf):
        return sample_stratified(tn, tf, self.num_bins)
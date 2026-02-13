from abc import ABC, abstractmethod
from typing import Callable, NamedTuple

import torch

import aurora.geometry as gmt


class RaySamples(NamedTuple):
    p: torch.Tensor
    t: torch.Tensor

class RaySampler(ABC):
    @abstractmethod
    def __call__(
        self,
        ro: torch.Tensor,
        rd: torch.Tensor,
        tn: torch.Tensor,
        tf: torch.Tensor
    ) -> RaySamples:
        ...


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

class EqualSampler(RaySampler):
    """Samples points at equal distances along rays."""
    def __init__(self, num_samples: int):
        super().__init__()
        self.num_samples = num_samples
    
    def __call__(self, ro, rd, tn, tf):
        t = sample_equal(tn, tf, self.num_samples)
        p = gmt.get_ray_points(ro, rd, t)
        return RaySamples(p, t)


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

class StratifiedSampler(RaySampler):
    """Samples random points each within same-sized bins along rays."""
    def __init__(self, num_bins: int):
        super().__init__()
        self.num_bins = num_bins
    
    def __call__(self, ro, rd, tn, tf):
        t = sample_stratified(tn, tf, self.num_bins)
        p = gmt.get_ray_points(ro, rd, t)
        return RaySamples(p, t)


def sample_pdf(
    bin_edges: torch.Tensor,
    weights: torch.Tensor,
    num_samples: int,
    normalize: bool = True,
    sort: bool = True
) -> torch.Tensor:
    """
    Samples `num_samples` values per batch element from a piecewise-constant PDF.

    The PDF is constant within each bin, with the bin's weight taken as the
    average of its two edge weights: bin_w[k] = 0.5 * (w[k] + w[k+1]).

    Args:
        bin_edges: (B, N+1) increasing bin boundaries.
        weights: Positive weights at each edge if shape (B, N+1) or within a bin if (B, N).
        num_samples: Number of samples to draw per batch element.
        normalize: Normalize weights if they don't sum to 1.

    Returns:
        (B, num_samples) samples, each in [bin_edges[b,0], bin_edges[b,-1]]
    """
    B, Np1 = weights.shape
    N = Np1 - 1
    device = weights.device

    # Bin weights as normalized average of weights at edges
    if weights.size(-1) == Np1:
        bin_weights = 0.5 * (weights[:, :-1] + weights[:, 1:]) # (B, N)
        bin_weights = bin_weights + 1e-10 # avoid zero
    if normalize:
        bin_weights = bin_weights / bin_weights.sum(dim=-1, keepdim=True)

    # CDF at bin edges: 0 at the left, 1 at the right
    cdf = torch.zeros(B, N + 1, device=device)
    cdf[:, 1:] = torch.cumsum(bin_weights, dim=-1)
    cdf = torch.clamp(cdf, 0.0, 1.0)

    # Sample uniform values and find their bins via searchsorted
    u = torch.rand(B, num_samples, device=device)
    inds = torch.searchsorted(cdf.contiguous(), u.contiguous(), right=True)
    inds = torch.clamp(inds, 1, N)  # right-edge index in [1, N]
    k = inds - 1 # bin index in [0, N-1]

    # Linear interpolation within the bin (uniform density => linear CDF)
    e_lo = bin_edges[:, :-1].gather(1, k) # (B, num_samples)
    e_hi = bin_edges[:, 1: ].gather(1, k)
    cdf_lo = cdf[:, :-1].gather(1, k)
    cdf_hi = cdf[:, 1: ].gather(1, k)

    denom = torch.clamp(cdf_hi - cdf_lo, min=1e-10)
    frac = (u - cdf_lo) / denom # position within bin [0, 1]
    t = e_lo + frac * (e_hi - e_lo)

    if sort:
        # Sort by distance (for integration)
        t, _ = torch.sort(t, dim=-1)

    return t  # (B, num_samples)

class HierarchicalSampler(RaySampler):
    """
    Samples points according to a weight distribution provided by another sampler.
    """
    def __init__(
            self,
            coarse_sampler: RaySampler,
            weight_fn: Callable[[torch.Tensor], torch.Tensor],
            num_samples: int,
            combine_samples: bool = False
        ):
        """
        Args:
            coarse_sampler: Coarse sampler used to sample the points along the
                rays where the weight will be evaluated.
            weight_fn: Function that maps 3D points tensor to a scalar weight
                tensor to be used as a PDF.
            num_samples: Number of fine samples to draw.
            combine_samples: Weither to combine the coarse and fine samples.
        """
        super().__init__()
        self.coarse_sampler = coarse_sampler
        self.weight_fn = weight_fn
        self.num_samples = num_samples
        self.combine_samples = combine_samples

    def __call__(self, ro, rd, tn, tf):
        pc, tc = self.coarse_sampler(ro, rd, tn, tf) # (B, Nc, 3), (B, Nc)
        pc = pc.flatten(0, 1) # (B*Nc, 3)
        w = self.weight_fn(pc) # (B*Nc,)
        w = w.reshape_as(tc) # (B, Nc)
        sort = not self.combine_samples
        t = sample_pdf(tc, w, self.num_samples, sort=sort) # (B, Nf)
        p = gmt.get_ray_points(ro, rd, t) # (B, Nf, 3)
        if self.combine_samples:
            t = torch.cat([t, tc], dim=1) # (B, Nc + Nf)
            p = torch.cat([p, pc], dim=1) # (B, Nc + Nf, 3)
            t, i = torch.sort(t, dim=1)
            p = torch.take_along_dim(p, i.unsqueeze(-1), dim=1)
        return RaySamples(p, t)
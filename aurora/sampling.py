import torch


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
    
    t01 = torch.linspace(0.0, 1.0, num_samples, device=device) # (num_samples,)
    t01 = t01.expand(n, -1) # (n, num_samples)
    
    t_min = t_min.unsqueeze(-1) # (n, 1)
    t_max = t_max.unsqueeze(-1) # (n, 1)
    t = t_min + t01 * (t_max - t_min) # (n, num_samples)
    return t

def sample_random_in_bins(
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
    t01 = torch.rand(n, num_bins, device=device) # (n, num_bins)
    t = lower_edges + t01 * (upper_edges - lower_edges) # (n, num_bins)
    return t

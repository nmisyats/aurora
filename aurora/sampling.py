import torch


def sample_uniform(
        t_min: torch.Tensor,
        t_max: torch.Tensor,
        num_samples: int
    ):
    """
    Samples equidistant `num_samples` samples between [min_t, max_t], edges included.
    
    Args:
        min_t: (n,) tensor representing starting points.
        max_t: (n,) tensor representing end points.
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
    Splits [min_t, max_t] into `num_bins` uniform bins and samples randomly in
    each bin.
    
    Args:
        min_t: (n,) tensor representing starting points.
        max_t: (n,) tensor representing end points.
        num_bins: Number of samples of [min_t, max_t].

    Returns:
        (n, num_samples) tensor.
    """
    device = t_min.device
    n = t_min.shape[0]
    
    t01 = torch.rand(n, num_bins, device=device) # (n, num_samples)
    
    t_min = t_min.unsqueeze(-1) # (n, 1)
    t_max = t_max.unsqueeze(-1) # (n, 1)
    t = t_min + t01 * (t_max - t_min) # (n, num_samples)
    return t

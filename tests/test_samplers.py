import torch

from aurora.samplers import sample_pdf


def test_sample_pdf_gaussian():
    torch.manual_seed(0)

    B = 4
    N = 128 # number of bins (N+1 edges)
    num_samples = 10_000
    mu, sigma = 0.0, 1.0
    x_min, x_max = -4.0, 4.0

    # Build bin edges and evaluate a Gaussian at each edge
    edges = torch.linspace(x_min, x_max, N + 1).unsqueeze(0).expand(B, -1)  # (B, N+1)
    weights = torch.exp(-0.5 * ((edges - mu) / sigma) ** 2) # (B, N+1)

    t = sample_pdf(edges, weights, num_samples) # (B, num_samples)

    assert t.shape == (B, num_samples), f"Unexpected shape: {t.shape}"
    assert t.min() >= x_min - 1e-5
    assert t.max() <= x_max + 1e-5

    # Count samples in a few equal-probability intervals of N(mu, sigma)
    # and check each bin gets roughly its expected share (num_samples / num_bins).
    # Equal-probability cuts for N(0,1): split at -0.675, 0, +0.675
    # giving four bins each with ~25% probability mass.
    cuts = torch.tensor([-0.675, 0.0, 0.675])
    expected = num_samples / 4
    tol = 0.05 * num_samples # allow 5% absolute deviation

    for b in range(B):
        s = t[b]
        counts = torch.tensor([
            (s <  cuts[0]).sum(),
            ((s >= cuts[0]) & (s < cuts[1])).sum(),
            ((s >= cuts[1]) & (s < cuts[2])).sum(),
            (s >= cuts[2]).sum(),
        ]).float()
        max_deviation = (counts - expected).abs().max().item()
        assert max_deviation < tol, (
            f"Batch {b}: bin counts {counts.tolist()} deviate too far from "
            f"expected {expected:.0f} (max deviation {max_deviation:.0f} >= tol {tol:.0f})."
        )
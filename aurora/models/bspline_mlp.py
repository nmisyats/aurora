import torch
import torch.nn as nn

from aurora.models.flux_model import FluxModel, ModelConfig
from aurora.frame import Frame
from aurora.bbox import BBox
import aurora.dnn as ann


class BSplineMLP(FluxModel):
    """MLP learning a B-spline basis of the flux"""
    def __init__(
            self,
            config: ModelConfig,
            encoding_exp: int = 4,
            max_log_flux: float = 7.0,
            num_basis: int = 8,
            num_hidden: int = 4,
            hidden_size: int =  128
        ):
        super().__init__(config)

        self.num_basis = num_basis
        self.max_log_flux = max_log_flux

        self.encoder = ann.FourierEncoder(encoding_exp)

        encode_dim = self.encoder.output_dim(2)
        hidden_sizes = [hidden_size] * num_hidden
        self.mlp = ann.MLP(encode_dim, self.num_basis, hidden_sizes)
        
        # Pre-compute basis functions
        self.register_buffer('basis_matrix_T', self._create_basis_matrix().T)

    def _create_basis_matrix(self) -> torch.Tensor:
        x = torch.log(self.energy_bins)
        x = (x - x.min()) / (x.max() - x.min())
        if x.ndim != 1:
            raise ValueError(f"x must be 1D (shape (P,)), got shape {tuple(x.shape)}")
        
        N = self.num_basis
        if N < 4:
            raise ValueError("For cubic (degree 3) clamped B-splines, need N >= 4.")

        device, dtype = x.device, x.dtype
        p = 3

        a = x.min()
        b = x.max()
        if torch.isclose(a, b):
            raise ValueError("All x are (almost) identical; cannot build a meaningful knot vector.")

        # Open-uniform knots over [a,b]:
        # len(knots) = N + p + 1 = N + 4
        # first 4 knots = a, last 4 knots = b, interior knots count = N - 4
        M = N + p + 1  # number of knots
        num_spans = N - p  # = N-3, number of uniform intervals
        h = (b - a) / num_spans

        # Interior knots: a+h, a+2h, ..., a+(N-4)h  (count N-4)
        if N - 4 > 0:
            interior = a + h * torch.arange(1, N - 3, device=device, dtype=dtype)  # 1 .. (N-4)
            knots = torch.cat([a.repeat(4), interior, b.repeat(4)])
        else:
            knots = torch.cat([a.repeat(4), b.repeat(4)])

        # Avoid the x == b boundary causing all-zero basis due to half-open intervals
        eps = (b - a) * torch.tensor(1e-7, device=device, dtype=dtype)
        x_eval = torch.where(x == b, x - eps, x)

        # Degree-0 basis: B_{i,0}(x) = 1 if t_i <= x < t_{i+1}
        # Shape: (P, M-1)
        t0 = knots[:-1]          # (M-1,)
        t1 = knots[1:]           # (M-1,)
        B = ((x_eval[:, None] >= t0[None, :]) & (x_eval[:, None] < t1[None, :])).to(dtype)

        # Cox–de Boor recursion up to degree p=3
        # After d steps, basis count reduces by 1 each time.
        for d in range(1, p + 1):
            # New basis count:
            # B_prev has shape (P, M-d)
            # B_new  has shape (P, M-d-1)
            i_max = M - d - 1  # number of basis functions at degree d

            ti   = knots[0:i_max]                 # t_i
            tid  = knots[d:d + i_max]             # t_{i+d}
            ti1  = knots[1:1 + i_max]             # t_{i+1}
            tid1 = knots[d + 1:d + 1 + i_max]     # t_{i+d+1}

            left_denom = (tid - ti)
            right_denom = (tid1 - ti1)

            # Fractions, with safe handling of zero denominators (repeated knots)
            left_frac = torch.where(
                left_denom != 0, (x_eval[:, None] - ti[None, :]) / left_denom[None, :], torch.zeros_like(x_eval[:, None])
            )
            right_frac = torch.where(
                right_denom != 0, (tid1[None, :] - x_eval[:, None]) / right_denom[None, :], torch.zeros_like(x_eval[:, None])
            )

            B_left = left_frac * B[:, :i_max]
            B_right = right_frac * B[:, 1:i_max + 1]
            B = B_left + B_right

        # Now B has shape (P, N) because M - p - 1 = (N+4) - 3 - 1 = N
        return B
    
    def forward(self, xy: torch.Tensor):
        xy = self.bbox.norm_xy(xy)
        xy_enc = self.encoder(xy)
        coeffs = self.mlp(xy_enc) # (batch_size, num_basis)
        log_f_at_edges = torch.matmul(coeffs, self.basis_matrix_T) # (batch_size, num_edges)
        log_f_at_edges *= self.max_log_flux
        f_at_edges = torch.pow(10.0, log_f_at_edges)
        f = 0.5 * (f_at_edges[:, :-1] + f_at_edges[:, 1:])
        f = torch.max(f, torch.tensor(1e-3, device=f.device))
        log_f = torch.log(f)
        return {
            "f": f,
            "f_at_edges": f_at_edges,
            "log_f_at_edges": log_f_at_edges,
            "log_f": log_f,
            "coeffs": coeffs
        }

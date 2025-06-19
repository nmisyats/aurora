import torch
import torch.nn as nn

class FourierEncoder(nn.Module):
    """Fourier feature encoding for positional information"""
    def __init__(self, encoding_exp: int):
        """
        Args:
            input_dim: Dimension of input (e.g., 2 for xy, 3 for xyE)
            encoding_exp: Number of frequency levels (2^0, 2^1, ..., 2^(encoding_exp-1))
        """
        assert encoding_exp >= 1, "encoding_exp must be at least 1"

        super().__init__()
        
        self.encoding_exp = encoding_exp
        
        # Pre-compute frequency multipliers
        freqs = [(2**i) * torch.pi for i in range(encoding_exp)]
        self.register_buffer("freqs", torch.tensor(freqs))
    
    def output_dim(self, input_dim: int):
        """Returns the encoded size of a tensor of size input_dim"""
        return (2 * self.encoding_exp + 1) * input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding to input tensor
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Encoded tensor of shape (..., output_dim)
        """
        # Move frequencies to same device as input
        
        # Compute sin/cos features efficiently
        # x: (..., input_dim), freqs: (encoding_exp,)
        # Result: (..., input_dim, encoding_exp)
        freq_x = self.freqs.unsqueeze(0).unsqueeze(-1) * x.unsqueeze(-2)
        
        cos_features = torch.cos(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        sin_features = torch.sin(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        
        # Concatenate original input with sin/cos features
        return torch.cat([x, cos_features, sin_features], dim=-1)
    
    def __repr__(self):
        return f"FourierEncoder(encoding_exp={self.encoding_exp})"

class Exponentiate(nn.Module):
    def __init__(
            self,
            base: float = None,
            log_min: float | None = None,
            log_max: float | None = None
        ):
        super().__init__()

        self.base = base
        self.log_min = log_min
        self.log_max = log_max

    def forward(self, x: torch.Tensor):
        x = torch.clamp(x, min=self.log_min, max=self.log_max)
        if self.base is None:
            exp_x = torch.exp(x)
        else:
            exp_x = torch.pow(self.base, x)
        return exp_x
    
    def __repr__(self):
        return (
            f"Exponentiate("
            f"base={self.base if self.base else 'e'}, "
            f"log_min={self.log_min}, "
            f"log_max={self.log_max}"
            f")"
        )
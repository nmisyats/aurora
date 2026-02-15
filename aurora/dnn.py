import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEncoder(nn.Module):
    """Fourier feature encoding for positional information"""
    def __init__(self, encoding_exp: int):
        """
        Args:
            encoding_exp: Number of frequency levels (2^0, 2^1, ..., 2^(encoding_exp-1))
        """
        super().__init__()
        
        self.encoding_exp = encoding_exp
        
        # Pre-compute frequency multipliers
        freqs = [(2**i) * torch.pi for i in range(encoding_exp)]
        self.register_buffer("freqs", torch.tensor(freqs))
    
    def output_dim(self, input_dim: int):
        """Returns the encoded size of a tensor of size input_dim"""
        return 2 * self.encoding_exp * input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding to input tensor
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Encoded tensor of shape (..., output_dim)
        """
        if x.dim() == 1:
            return self._forward_single(x)
        else:
            return self._forward_batch(x)
    
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_batch(x.unsqueeze(0)).squeeze(0)
    
    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
        # Compute sin/cos features efficiently
        # x: (..., input_dim), freqs: (encoding_exp,)
        # Result: (..., input_dim, encoding_exp)
        freq_x = self.freqs.unsqueeze(0).unsqueeze(-1) * x.unsqueeze(-2)
        
        cos_features = torch.cos(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        sin_features = torch.sin(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        
        # Concatenate original input with sin/cos features
        return torch.cat([sin_features, cos_features], dim=-1)
    
    def __repr__(self):
        return f"FourierEncoder(encoding_exp={self.encoding_exp})"


class MLP(nn.Sequential):
    """Multi-layer perceptron (MLP) neural network with inner ReLU activations"""
    def __init__(self, in_dim: int, out_dim: int, hidden_dims=()):
        """Creates a multi-layer perceptron (MLP).
 
        Args:
            in_dim: Input feature dimension.
            out_dim: Output feature dimension.
            hidden_dims: Hidden layer sizes. If empty, the MLP is a single Linear
                layer mapping `in_dim -> out_dim`.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims

        sizes = (in_dim, *hidden_dims, out_dim)
        for i in range(len(sizes) - 2):
            self.append(nn.Linear(sizes[i], sizes[i + 1]))
            self.append(nn.ReLU())
        self.append(nn.Linear(sizes[-2], sizes[-1]))

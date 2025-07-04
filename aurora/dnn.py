from typing import Optional

import torch
import torch.nn as nn

class FourierEncoder(nn.Module):
    """Fourier feature encoding for positional information"""
    def __init__(self, encoding_exp: int):
        """
        Args:
            encoding_exp: Number of frequency levels (2^0, 2^1, ..., 2^(encoding_exp-1)).
                Setting encoding_exp = 0 is equivalent to identity.
        """
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
        if x.dim() == 1:
            return self._forward_single(x)
        else:
            return self._forward_batch(x)
    
    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_batch(x.unsqueeze(0)).squeeze(0)
    
    def _forward_batch(self, x: torch.Tensor) -> torch.Tensor:
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

def clamped_exp10(
        x: torch.Tensor,
        log_min: Optional[float] = None,
        log_max: Optional[float] = None
    ):
    """Computes 10^x with optional clamping of the input values.
    
    Args:
        x: Input tensor containing the exponent values.
        log_min: Optional minimum value to clamp x to. If None, no lower bound.
        log_max: Optional maximum value to clamp x to. If None, no upper bound.
        
    Returns:
        torch.Tensor: 10^x where x has been clamped to the specified range
            [log_min, log_max].
    """
    x = torch.clamp(x, log_min, log_max)
    return torch.pow(10.0, x)

def make_mlp(*sizes: int):
    """Creates a multi-layer perceptron (MLP) neural network with inner ReLU activations.
    
    Args:
        *sizes: Variable number of integers specifying the layer sizes.
               Must provide at least 2 sizes (input and output dimensions).
               
    Returns:
        nn.Module: A PyTorch Sequential module containing the MLP layers.
                  For 2 sizes, returns a single Linear layer.
                  For 3+ sizes, returns a Sequential with Linear layers and ReLU 
                  activations between hidden layers (no activation after final layer).
                  
    Raises:
        AssertionError: If fewer than 2 sizes are provided.
    """
    assert len(sizes) >= 2

    if len(sizes) == 2:
        return nn.Linear(sizes[0], sizes[1])
    
    mlp = nn.Sequential()
    num_hidden = len(sizes) - 2
    for i in range(num_hidden):
        mlp.append(nn.Linear(sizes[i], sizes[i+1]))
        mlp.append(nn.ReLU())
    mlp.append(nn.Linear(sizes[-2], sizes[-1]))
    
    return mlp
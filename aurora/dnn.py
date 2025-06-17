import torch

class FourierEncoder:
    """Fourier feature encoding for positional information"""
    def __init__(self, input_dim: int, encoding_exp: int):
        """
        Args:
            input_dim: Dimension of input (e.g., 2 for xy, 3 for xyE)
            encoding_exp: Number of frequency levels (2^0, 2^1, ..., 2^(encoding_exp-1))
        """
        assert encoding_exp >= 1, "encoding_exp must be at least 1"
        
        self.input_dim = input_dim
        self.encoding_exp = encoding_exp
        self.output_dim = (2 * encoding_exp + 1) * input_dim
        
        # Pre-compute frequency multipliers
        self.freqs = torch.tensor([(2**i) * torch.pi for i in range(encoding_exp)])
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding to input tensor
        
        Args:
            x: Input tensor of shape (..., input_dim)
            
        Returns:
            Encoded tensor of shape (..., output_dim)
        """
        # Move frequencies to same device as input
        freqs = self.freqs.to(x.device)
        
        # Compute sin/cos features efficiently
        # x: (..., input_dim), freqs: (encoding_exp,)
        # Result: (..., input_dim, encoding_exp)
        freq_x = freqs.unsqueeze(0).unsqueeze(-1) * x.unsqueeze(-2)
        
        cos_features = torch.cos(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        sin_features = torch.sin(freq_x).flatten(start_dim=-2)  # (..., input_dim * encoding_exp)
        
        # Concatenate original input with sin/cos features
        return torch.cat([x, cos_features, sin_features], dim=-1)
    
    def __repr__(self):
        return f"FourierEncoder(input_dim={self.input_dim}, encoding_exp={self.encoding_exp}, output_dim={self.output_dim})"

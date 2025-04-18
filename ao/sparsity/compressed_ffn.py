import torch
import torch.nn as nn
from typing import Optional, Tuple
from .activation_compression import CompressedActivation

class SquaredReLU(nn.Module):
    """Squared ReLU activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x) ** 2

class CompressedFFN(nn.Module):
    """
    Feed-forward network with SquaredReLU activation and compression.
    This implementation follows the paper's approach for high activation sparsity.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        compression_method: str = 'simple'
    ):
        """
        Initialize the compressed FFN.
        
        Args:
            d_model: Input/output dimension
            d_ff: Hidden dimension of the feed-forward network
            dropout: Dropout probability
            compression_method: Method to use for activation compression
        """
        super().__init__()
        
        # First linear layer
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        
        # Activation and compression
        self.activation = SquaredReLU()
        self.compression = CompressedActivation(compression_method)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights for better sparsity
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to promote sparsity."""
        # Initialize with smaller weights to promote sparsity
        nn.init.normal_(self.w1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.w2.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with activation compression.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # First linear transformation
        x = self.w1(x)
        
        # Apply activation and compression
        x = self.activation(x)
        
        # Only compress if we're in training mode
        if self.training:
            x = self.compression(x)
        
        # Second linear transformation
        x = self.w2(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x
    
    def get_compression_stats(self) -> Tuple[float, float]:
        """
        Get statistics about the compression.
        
        Returns:
            Tuple containing:
            - Compression ratio (original_size / compressed_size)
            - Sparsity ratio (number of zeros / total elements)
        """
        if not self.training or self.compression.compressed_data is None:
            return 1.0, 0.0
            
        # Calculate original size
        original_size = (
            self.compression.compressed_data['shape'][0] *
            self.compression.compressed_data['shape'][1] *
            self.compression.compressed_data['shape'][2]
        )
        
        # Calculate compressed size
        compressed_size = (
            self.compression.compressed_data['values'].numel() +
            self.compression.compressed_data['indices'].numel()
        )
        
        # Calculate sparsity
        sparsity = 1.0 - (self.compression.compressed_data['values'].numel() / original_size)
        
        return original_size / compressed_size, sparsity 
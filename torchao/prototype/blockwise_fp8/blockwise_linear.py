import torch
from torch import nn
from typing import Optional
from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm 
from torchao.prototype.blockwise_fp8.blockwise_quantization import fp8_blockwise_act_quant

def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, block_size = 128) -> torch.Tensor:
    x, scale = fp8_blockwise_act_quant(x, block_size)
    y = blockwise_fp8_gemm(x, scale, weight, weight.scale)
    if bias is not None:
        y += bias
    return y

class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, block_size = 128, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size
            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)
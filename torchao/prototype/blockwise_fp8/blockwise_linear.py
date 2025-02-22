import torch
from torch import nn

from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm
from torchao.prototype.blockwise_fp8.blockwise_quantization import (
    fp8_blockwise_act_quant,
)


class BlockwiseQuantLinear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, block_size:int = 128, dtype = torch.float8_e4m3fn):
        super().__init__()
        assert dtype is torch.float8_e4m3fn, "Only float8_e4m3fn is supported for now."
        scale_in_features = (in_features + block_size - 1) // block_size
        scale_out_features = (out_features + block_size - 1) // block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        self.block_size = block_size

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
        x, scale = fp8_blockwise_act_quant(x, self.block_size)
        y = blockwise_fp8_gemm(x, scale, self.weight, self.weight.scale)
        
        if self.bias is not None:
            y += self.bias
        return y
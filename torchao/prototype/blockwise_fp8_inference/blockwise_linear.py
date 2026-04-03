# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchao.float8.config import e4m3_dtype
from torchao.kernel.blockwise_quantization import (
    blockwise_fp8_gemm,
    fp8_blockwise_act_quant,
)

_SUPPORTED_FP8_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
]


class BlockwiseQuantLinear(nn.Module):
    """
    Custom linear layer with support for blockwise FP8 quantized weights.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
        dtype (torch.dtype): FP8 data type for quantized weights.
            Defaults to the hardware-appropriate e4m3 variant
            (e4m3fn on NVIDIA, e4m3fnuz on MI300).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 128,
        dtype: torch.dtype = e4m3_dtype,
    ):
        super().__init__()
        assert dtype in _SUPPORTED_FP8_DTYPES, (
            f"Unsupported dtype: {dtype}. Supported dtypes: {_SUPPORTED_FP8_DTYPES}"
        )
        scale_in_features = (in_features + block_size - 1) // block_size
        scale_out_features = (out_features + block_size - 1) // block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        self.weight.scale = self.scale = nn.Parameter(
            torch.empty(scale_out_features, scale_in_features, dtype=torch.float32)
        )
        self.block_size = block_size
        self.fp8_dtype = dtype

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
        x, scale = fp8_blockwise_act_quant(x, self.block_size, self.fp8_dtype)
        y = blockwise_fp8_gemm(
            x, scale, self.weight, self.weight.scale, self.block_size
        )

        if self.bias is not None:
            y += self.bias
        return y

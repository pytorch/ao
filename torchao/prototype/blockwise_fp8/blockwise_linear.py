# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.blockwise_fp8.deep_gemm_utils import (
    scaled_mm_deep_gemm_128_1_128_1,
    scaled_mm_deep_gemm_128_1_128_128,
)
from torchao.prototype.blockwise_fp8.kernels import (
    fp8_blockwise_act_quant,
    triton_quantize_fp8_block,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)


class fp8_blockwise_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size):
        assert block_size == 128, "Only support block_size=128"

        # Temporarily reshape x to 2D tensor
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Triton kernel from DeepGEMM currently has the fastest activation quantization (1 x block_size)
        x_fp8, x_scale = fp8_blockwise_act_quant(x, block_size)

        # fbgemm currently has the fastest weight quantization (block_size x block_size)
        weight_t_fp8, weight_t_scale = triton_quantize_fp8_block(
            weight,
            block_m=block_size,
            block_k=block_size,
            k_major=True,  # For [M,K] -> [K,M] in column-major
        )

        # DeepGEMM for blockwise GEMM where activation has (1 x block_size) scaling granularity
        # and weight has (block_size x block_size) scaling granularity.
        out = scaled_mm_deep_gemm_128_1_128_128(
            x_fp8,
            x_scale,
            weight_t_fp8,
            weight_t_scale,
        )
        ctx.save_for_backward(x, weight)
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size

        # left operand must be row-major
        grad_output_fp8, grad_output_scale = fp8_blockwise_act_quant(
            grad_output,
            block_size,
        )

        # right operand must be column-major
        weight_t_fp8, weight_t_scale = triton_quantize_fp8_block(
            weight,
            block_m=block_size,
            block_k=block_size,
            k_major=False,  # For [M,K] -> [K,M] in row-major
        )
        weight_t_fp8 = weight_t_fp8.t().contiguous().t()  # To col-major

        # DeepGEMM for blockwise GEMM where left operand has (1 x block_size) scaling granularity
        # and right operand has (block_size x block_size) scaling granularity.
        # grad_x = grad_output @ weight.T
        grad_x = scaled_mm_deep_gemm_128_1_128_128(
            grad_output_fp8,
            weight_t_fp8,
            1.0 / grad_output_scale,
            1.0 / weight_t_scale,
        )

        # left operand must be row-major
        grad_output_t_fp8, grad_output_t_scale = fp8_blockwise_act_quant(
            grad_output.t().contiguous(),
            block_size,
        )

        # right operand must be column-major
        x_fp8, x_scale = fp8_blockwise_act_quant(
            x,
            block_size,
        )
        x_fp8 = x_fp8.t().contiguous().t()  # To col-major

        # DeepGEMM for blockwise GEMM where both operands have (1 x block_size) scaling granularity.
        # grad_weight = grad_output.T @ x
        grad_weight = scaled_mm_deep_gemm_128_1_128_1(
            grad_output_t_fp8,
            x_fp8,
            1.0 / grad_output_t_scale,
            1.0 / x_scale,
        )
        return grad_x, grad_weight, None, None


class Float8BlockwiseLinear(nn.Linear):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        block_size (int): Block size for quantization. Defaults to 128.
        dtype (torch.dtype): Data type for the weights. Defaults to torch.float8_e4m3fn.
    """

    supported_dtypes = [
        torch.bfloat16,
    ]

    def __init__(
        self,
        *args,
        block_size: int = 128,
        dtype=torch.bfloat16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert dtype in self.supported_dtypes, (
            f"Unsupported dtype: {dtype}. Supported dtypes: {self.supported_dtypes}"
        )
        self.block_size = block_size
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return fp8_blockwise_mm.apply(x, self.weight, self.block_size)

    @classmethod
    def from_float(
        cls,
        mod,
    ):
        assert mod.bias is None, "unsupported"
        assert mod.in_features % 128 == 0, "unsupported"
        assert mod.out_features % 128 == 0, "unsupported"
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8BlockwiseLinearConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(Float8BlockwiseLinearConfig)
def _deep_gemm_float8_inference_linear_transform(module, config):
    return Float8BlockwiseLinear.from_float(module)

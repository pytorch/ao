# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.blockwise_fp8_training.kernels import (
    blockwise_fp8_gemm_1x128_128x1,
    blockwise_fp8_gemm_1x128_128x128,
    fp8_blockwise_act_quant_lhs,
    fp8_blockwise_act_quant_rhs,
    fp8_blockwise_act_quant_transposed_lhs,
    fp8_blockwise_weight_quant_rhs,
    fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import is_sm_at_least_90


class fp8_blockwise_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size):
        assert block_size == 128, "Only support block_size=128"

        # Temporarily reshape x to 2D tensor
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Cast inputs to fp8 blockwise using (1, block_size) scaling granularity in row major format.
        x_fp8, x_scale = fp8_blockwise_act_quant_lhs(x, block_size)

        # Cast weight to fp8 blockwise using (block_size, block_size) scaling granularity, with transposed dims in column major format.
        weight_t_fp8, weight_t_scale = fp8_blockwise_weight_quant_transposed_rhs(
            weight,
            block_size=block_size,
        )

        # out = input @ weight.T
        out = blockwise_fp8_gemm_1x128_128x128(
            x_fp8,
            1.0 / x_scale,
            weight_t_fp8,
            1.0 / weight_t_scale,
        )
        out = out.reshape(*x_orig_shape[:-1], out.shape[-1])
        ctx.save_for_backward(x, weight)
        ctx.block_size = block_size
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size

        # Reshape input to 2D
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Reshape grad_output to 2D
        grad_output_orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output_orig_shape[-1]).contiguous()
        assert grad_output.shape[1] % 128 == 0, "unsupported"

        # Cast grad_output to fp8 blockwise 1x128 since it is the grad of the output activation.
        grad_output_fp8, grad_output_scale = fp8_blockwise_act_quant_lhs(
            grad_output,
            block_size,
        )

        # Cast weight to fp8 blockwise to 128x128 in column major format.
        weight_fp8, weight_scale = fp8_blockwise_weight_quant_rhs(
            weight,
            block_size=block_size,
        )

        # grad_x = grad_output @ weight
        grad_x = blockwise_fp8_gemm_1x128_128x128(
            grad_output_fp8,
            1.0 / grad_output_scale,
            weight_fp8,
            1.0 / weight_scale,
        )

        # Cast grad_output_t to fp8 blockwise with (1 x block_size) scaling groups, since it is
        # the grad of the output activation.
        # Write directly with transposed dims in row major format, as needed for dW calc.
        grad_output_t_fp8, grad_output_t_scale = fp8_blockwise_act_quant_transposed_lhs(
            grad_output,
            block_size,
        )

        # Cast x to fp8 blockwise with (block_size x 1) scaling groups, in column major format.
        # RHS should have groupwise scales calculated colwise, so scaling groups do not cross the
        # contracting (K) dim.
        x_fp8, x_scale = fp8_blockwise_act_quant_rhs(x, block_size)

        # grad_weight = grad_output.T @ x
        grad_weight = blockwise_fp8_gemm_1x128_128x1(
            grad_output_t_fp8,
            1.0 / grad_output_t_scale,
            x_fp8,
            1.0 / x_scale,
        )

        # Reshape grad_x to expected potentially 3D+ shape
        grad_x = grad_x.reshape(*grad_output_orig_shape[:-1], grad_x.shape[-1])
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
        assert is_sm_at_least_90(), "Only support SM90"
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
def _float8_blockwise_transform(module, config):
    return Float8BlockwiseLinear.from_float(module)

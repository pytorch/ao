# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed._tensor import DTensor

from torchao.core.config import AOBaseConfig
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _prepare_blockwise_scaled_mm_rhs_scale,
    _scaling_type_value,
    blockwise_scaled_mm,
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128,
)
from torchao.quantization.transform_module import (
    register_quantize_module_handler,
)
from torchao.utils import is_sm_at_least_90


def _scaled_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_recipe_a,
    scale_b: torch.Tensor,
    scale_recipe_b,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if not any(
        isinstance(tensor, DTensor) for tensor in (mat_a, mat_b, scale_a, scale_b)
    ):
        return F.scaled_mm(
            mat_a,
            mat_b,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=_prepare_blockwise_scaled_mm_rhs_scale(scale_b, scale_recipe_b),
            scale_recipe_b=scale_recipe_b,
            output_dtype=out_dtype,
        )

    return blockwise_scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        _scaling_type_value(scale_recipe_a),
        scale_b,
        _scaling_type_value(scale_recipe_b),
        out_dtype,
    )


def _run_blockwise_mm(
    *,
    use_triton: bool,
    triton_kernel,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_recipe_a,
    scale_b: torch.Tensor,
    scale_recipe_b,
    triton_scale_b: torch.Tensor | None = None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if use_triton:
        return triton_kernel(
            mat_a,
            mat_b,
            scale_a,
            scale_b if triton_scale_b is None else triton_scale_b,
            out_dtype=out_dtype,
        )
    return _scaled_mm(
        mat_a,
        mat_b,
        scale_a,
        scale_recipe_a,
        scale_b,
        scale_recipe_b,
        out_dtype,
    )


class fp8_blockwise_mm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, block_size, out_dtype=torch.bfloat16, use_triton=False):
        assert block_size == 128, "Only support block_size=128"

        # Temporarily reshape x to 2D tensor
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Cast inputs to fp8 blockwise using (1, block_size) scaling granularity in row major format.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_lhs(x, block_size)

        # Cast weight to fp8 blockwise using (block_size, block_size) scaling granularity, with transposed dims in column major format.
        weight_t_fp8, weight_t_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
            weight,
            block_size=block_size,
        )

        # out = input @ weight.T
        out = _run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=triton_fp8_gemm_1x128_128x128,
            mat_a=x_fp8,
            mat_b=weight_t_fp8,
            scale_a=x_scale,
            scale_recipe_a=BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=weight_t_scale,
            scale_recipe_b=BLOCKWISE_128X128_SCALING_TYPE,
            out_dtype=out_dtype,
        )
        out = out.reshape(*x_orig_shape[:-1], out.shape[-1])
        ctx.save_for_backward(x, weight)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype
        ctx.use_triton = use_triton
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        use_triton = ctx.use_triton

        # Reshape input to 2D
        x_orig_shape = x.shape
        x = x.reshape(-1, x_orig_shape[-1])

        # Reshape grad_output to 2D
        grad_output_orig_shape = grad_output.shape
        grad_output = grad_output.reshape(-1, grad_output_orig_shape[-1]).contiguous()
        assert grad_output.shape[1] % 128 == 0, "unsupported"

        # Cast grad_output to fp8 blockwise 1x128 since it is the grad of the output activation.
        grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
            grad_output,
            block_size,
        )

        # Cast weight to fp8 blockwise to 128x128 in column major format.
        weight_fp8, weight_scale = triton_fp8_blockwise_weight_quant_rhs(
            weight,
            block_size=block_size,
        )

        # grad_x = grad_output @ weight
        grad_x = _run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=triton_fp8_gemm_1x128_128x128,
            mat_a=grad_output_fp8,
            mat_b=weight_fp8,
            scale_a=grad_output_scale,
            scale_recipe_a=BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=weight_scale,
            scale_recipe_b=BLOCKWISE_128X128_SCALING_TYPE,
            out_dtype=out_dtype,
        )

        # Cast grad_output_t to fp8 blockwise with (1 x block_size) scaling groups, since it is
        # the grad of the output activation.
        # Write directly with transposed dims in row major format, as needed for dW calc.
        grad_output_t_fp8, grad_output_t_scale = (
            triton_fp8_blockwise_act_quant_transposed_lhs(
                grad_output,
                block_size,
            )
        )

        # Cast x to fp8 blockwise with (block_size x 1) scaling groups, in column major format.
        # RHS should have groupwise scales calculated colwise, so scaling groups do not cross the
        # contracting (K) dim.
        x_fp8, x_scale = triton_fp8_blockwise_act_quant_rhs(x, block_size)

        # grad_weight = grad_output.T @ x
        grad_weight = _run_blockwise_mm(
            use_triton=use_triton,
            triton_kernel=triton_fp8_gemm_1x128_128x1,
            mat_a=grad_output_t_fp8,
            mat_b=x_fp8,
            scale_a=grad_output_t_scale,
            scale_recipe_a=BLOCKWISE_1X128_SCALING_TYPE,
            scale_b=x_scale.transpose(-1, -2),
            scale_recipe_b=BLOCKWISE_1X128_SCALING_TYPE,
            # In the grad_weight path, Triton 1x128_128x1 expects RHS scales
            # in row-major layout, while scaled_mm expects the transposed scale
            # layout. Pass a separate Triton scale tensor so each backend gets
            # its required layout.
            triton_scale_b=x_scale,
            out_dtype=out_dtype,
        )

        # Reshape grad_x to expected potentially 3D+ shape
        grad_x = grad_x.reshape(*grad_output_orig_shape[:-1], grad_x.shape[-1])
        return grad_x, grad_weight, None, None, None


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
        use_triton=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        assert dtype in self.supported_dtypes, (
            f"Unsupported dtype: {dtype}. Supported dtypes: {self.supported_dtypes}"
        )
        assert is_sm_at_least_90(), "Only support SM90"
        self.block_size = block_size
        self.dtype = dtype
        self.use_triton = use_triton

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return fp8_blockwise_mm.apply(
            x, self.weight, self.block_size, self.dtype, self.use_triton
        )

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

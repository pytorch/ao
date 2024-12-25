# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

import torch

from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    _ToFloat8ConstrFunc,
)
from torchao.prototype.float8nocompile.kernels.fp8_dynamic_tensorwise import (
    triton_hp_tensor_to_float8_dynamic,
)

# avoid division by zero when calculating scale
# TODO: align this value with NVIDIA's assumptions (current value is a guess)
EPS = 1e-12


def hp_tensor_to_float8nocompile_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
) -> Float8Tensor:
    """
    Given a high precision tensor `hp_tensor`,
    scales `hp_tensor` dynamically and returns a `Float8Tensor` of the result.

    Args:
        hp_tensor: the tensor to convert
        float8_dtype: the float8 dtype to use
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        gemm_input_role: Defines the role of this tensor (input, weight or grad_output) in
          the 3 fwd/bwd gemms of linear
    """
    # TODO(danielvegamyhre): replace this torch implementation with custom triton kernel
    # torch.compile and eager show different numerics for 1.0 / float32,
    # upcast to float64 to ensure same numeric between compile and eager
    amax = torch.max(torch.abs(hp_tensor)).to(torch.float64)
    scale = torch.finfo(float8_dtype).max / torch.clamp(amax, min=EPS)
    scale = scale.to(torch.float32)  # scale must be fp32
    return _ToFloat8ConstrFunc.apply(
        hp_tensor,
        scale,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
        None,
    )


class Float8NoCompileConversionFunc(torch.autograd.Function):
    """
    A differentiable conversion to fp8.
    * forward: convert from high precision to float8
    * backward: pass the gradient without changes
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
        gemm_input_role: GemmInputRole,
    ):
        return triton_hp_tensor_to_float8_dynamic(
            tensor,
            float8_dtype,
            linear_mm_config,
            gemm_input_role,
        )

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None, None


class NoopFwToFloat8NoCompileBwDynamic(torch.autograd.Function):
    """
    A differentiable conversion to fp8.
    * forward: no-op
    * backward: convert to float8 with tensor-wise dynamic scaling
    """

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        float8_dtype: torch.dtype,
        linear_mm_config: LinearMMConfig,
    ):
        ctx.linear_mm_config = linear_mm_config
        ctx.target_dtype = float8_dtype
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        fp8_tensor = triton_hp_tensor_to_float8_dynamic(
            gradY,
            ctx.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
        )
        return fp8_tensor, None, None

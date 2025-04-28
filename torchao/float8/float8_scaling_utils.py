# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for scaling high precision tensors to float8.
"""

from typing import Optional

import torch

from torchao.float8.config import ScalingGranularity
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import (
    tensor_to_scale,
)


# TODO(danielvegamyhre): refactor to accept Float8LinearConfig directly
def hp_tensor_to_float8_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    device_mesh=None,
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    axiswise_dim: Optional[int] = None,
    round_scales_to_power_of_2: bool = False,
) -> Float8Tensor:
    """
    Given a high precision tensor `hp_tensor`,
    scales `hp_tensor` dynamically and returns a `Float8Tensor` of the result.

    Args:
        hp_tensor: the tensor to convert
        float8_dtype: the float8 dtype to use
        linear_mm_config: Defines the configuration for the scaled_mm for
          the 3 fwd/bwd gemms of linear
        reduce_amax: whether to reduce the max(abs(hp_tensor)) value across distributed ranks
        gemm_input_role: Defines the role of this tensor (input, weight or grad_output) in
          the 3 fwd/bwd gemms of linear
        scaling_granularity: Defines the scaling granularity
        axiswise_dim: if axiswise granularity is used, defines the dim to scale across
        round_scales_to_power_of_2: if true, round scaling factor down to the nearest power of 2.
    """
    scale = tensor_to_scale(
        hp_tensor,
        float8_dtype,
        reduce_amax,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
        round_scales_to_power_of_2,
    )
    return hp_tensor_and_scale_to_float8(
        hp_tensor,
        scale,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
        axiswise_dim,
    )


def get_maybe_axiswise_dim(
    axiswise_dim: int,
    scaling_granularity: ScalingGranularity,
) -> Optional[int]:
    """
    Convenience function which takes in an axiswise dim which is only relevant
    for axiswise scaing, and a scaling type.  The output is pass-through
    if scaling type is axiswise, and None otherwise.  This is done to keep the
    logic from choosing the axiswise dim out of the scaling function.
    """
    if scaling_granularity is ScalingGranularity.AXISWISE:
        return axiswise_dim
    return None


@torch._dynamo.allow_in_graph
class NoopFwToFloat8BwDynamic(torch.autograd.Function):
    """
    Forward: no-op
    Backward: convert to target float8 dtype with dynamic scaling
    """

    @staticmethod
    def forward(
        ctx,
        tensor,
        linear_mm_config: LinearMMConfig,
        target_dtype: torch.dtype,
        axiswise_dim: int,
    ):
        ctx.linear_mm_config = linear_mm_config
        ctx.target_dtype = target_dtype
        ctx.axiswise_dim = axiswise_dim
        return tensor

    @staticmethod
    def backward(ctx, gradY):
        if tensor_already_casted_to_fp8(gradY):
            return gradY, None, None
        gradY_scale = tensor_to_scale(gradY, ctx.target_dtype)
        fp8_tensor = hp_tensor_and_scale_to_float8(
            gradY,
            gradY_scale,
            ctx.target_dtype,
            ctx.linear_mm_config,
            GemmInputRole.GRAD_OUTPUT,
            axiswise_dim=ctx.axiswise_dim,
        )
        return fp8_tensor, None, None

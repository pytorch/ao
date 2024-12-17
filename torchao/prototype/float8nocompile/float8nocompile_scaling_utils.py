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
    hp_tensor_and_scale_to_float8,
    LinearMMConfig,
)
from torchao.float8.float8_utils import tensor_to_scale


def hp_tensor_to_float8nocompile_dynamic(
    hp_tensor: torch.Tensor,
    float8_dtype: torch.dtype,
    linear_mm_config: LinearMMConfig,
    reduce_amax: bool = False,
    gemm_input_role: GemmInputRole = GemmInputRole.INPUT,
    device_mesh=None,
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE,
    axiswise_dim: Optional[int] = None,
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
    """
    # TODO(danielvegamyhre): replace this torch implementation with custom triton kernel
    if tensor_already_casted_to_fp8(hp_tensor):
        return hp_tensor
    scale = tensor_to_scale(
        hp_tensor,
        float8_dtype,
        reduce_amax,
        device_mesh,
        scaling_granularity,
        axiswise_dim,
    )
    return hp_tensor_and_scale_to_float8(
        hp_tensor,
        scale,
        float8_dtype,
        linear_mm_config,
        gemm_input_role,
        axiswise_dim,
    )

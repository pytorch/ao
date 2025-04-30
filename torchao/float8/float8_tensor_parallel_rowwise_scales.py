# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from math import modf
import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from torchao.float8.config import ScalingGranularity, ScalingType, e4m3_dtype
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_scaling_utils import (
    NoopFwToFloat8BwDynamic,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig

# subclass the ColwiseParallel and RowwiseParallel classes
# to add the float8 support
# The parameter sharding stays the same as the core
# ColwiseParallel and RowwiseParallel, the only difference
# here is that in input/output handling we do casting after
# creating the DTensor.

# NOTE: This only works and tested with the dynamic scaling


def _float8_linear_supports_float8_allgather(m):
    # TODO(future PR): also gate this by granularity
    return (
        m.scaling_type_input == ScalingType.DYNAMIC
        and m.scaling_type_grad_output == ScalingType.DYNAMIC
    )


class Float8ColwiseParallel(ColwiseParallel):
    """
    Like `ColwiseParallel`, but with all-gather in float8 with rowwise scales.
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh,
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )
        if not tensor_already_casted_to_fp8(input_tensor):
            input_tensor_row_major_rowwise_scales = hp_tensor_to_float8_dynamic(
                input_tensor,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)

            # For columnwise scales, we have to reshape to 2D before scaling along dim=0,
            # otherwise scale will be along the batch dim, which isn't what we need.
            # We will reshape back to 3D after the all-gather in Float8Linear.forward()
            input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
            input_tensor_col_major_colwise_scales = hp_tensor_to_float8_dynamic(
                input_2d,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=0,
            )  # DTensor(Float8Tensor) 
        else:
            input_tensor_row_major_rowwise_scales = input_tensor if is_row_major(input_tensor) else None
            input_tensor_col_major_colwise_scales = input_tensor if not is_row_major(input_tensor) else None

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            if input_tensor_row_major_rowwise_scales is not None:
                input_tensor_row_major_rowwise_scales = input_tensor_row_major_rowwise_scales.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            if input_tensor_col_major_colwise_scales is not None:
                input_tensor_col_major_colwise_scales = input_tensor_col_major_colwise_scales.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
        return input_tensor_row_major_rowwise_scales, input_tensor_col_major_colwise_scales

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # outputs is a shard on last dimension DTensor, i.e. Shard(-1)
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(
                placements=output_layouts, async_op=True
            )  # DTensor(torch.Tensor)

        # do not convert output to float8
        
        # back to local tensor
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torchao.float8.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

        return super()._apply(module, device_mesh)


class Float8RowwiseParallel(RowwiseParallel):
    """
    Like `RowwiseParallel`, but with all-gather in float8 with rowwise scales.
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh,
    ):
        # annotate module input placements/sharding with input_layouts
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(
                input_tensor, device_mesh, input_layouts, run_check=False
            )
        if not tensor_already_casted_to_fp8(input_tensor):
            input_tensor_row_major_rowwise_scales = hp_tensor_to_float8_dynamic(
                input_tensor,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)

            # For columnwise scales, we have to reshape to 2D before scaling along dim=0,
            # otherwise scale will be along the batch dim, which isn't what we need.
            # We will reshape back to 3D after the all-gather in Float8Linear.forward()
            input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
            input_tensor_col_major_colwise_scales = hp_tensor_to_float8_dynamic(
                input_2d,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=0,
            )  # DTensor(Float8Tensor) 
        else:
            input_tensor_row_major_rowwise_scales = input_tensor if is_row_major(input_tensor) else None
            input_tensor_col_major_colwise_scales = input_tensor if not is_row_major(input_tensor) else None

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            if input_tensor_row_major_rowwise_scales is not None:
                input_tensor_row_major_rowwise_scales = input_tensor_row_major_rowwise_scales.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            if input_tensor_col_major_colwise_scales is not None:
                input_tensor_col_major_colwise_scales = input_tensor_col_major_colwise_scales.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
        return input_tensor_row_major_rowwise_scales, input_tensor_col_major_colwise_scales

    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)
        
        # do not convert output to float8

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torchao.float8.float8_linear import Float8Linear

        if not isinstance(module, Float8Linear):
            raise ValueError(
                f"Expecting module to be Float8Linear but found {type(module)}"
            )
        elif isinstance(
            module, Float8Linear
        ) and not _float8_linear_supports_float8_allgather(module):
            raise AssertionError("unsupported")

        return super()._apply(module, device_mesh)



def is_row_major(tensor: torch.Tensor) -> bool:
    assert tensor.dim() >= 2
    return tensor.stride(-2) > tensor.stride(-1) and tensor.stride(-1) == 1

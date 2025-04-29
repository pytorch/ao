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

            input_tensor_col_major_colwise_scales = hp_tensor_to_float8_dynamic(
                input_tensor,
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

        # outputs = NoopFwToFloat8RowwiseBwDynamic.apply(
        #     outputs,
        #     mod.linear_mm_config,
        #     mod.config.cast_config_grad_output.target_dtype,
        #     -1, # axiswise_dim
        # )
        # outputs = NoopFwToFloat8BwDynamic.apply(
        #     outputs,
        #     mod.linear_mm_config,
        #     mod.config.cast_config_grad_output.target_dtype,
        # )
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

            input_tensor_col_major_colwise_scales = hp_tensor_to_float8_dynamic(
                input_tensor.reshape(-1, input_tensor.shape[-1]),
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

        # outputs = NoopFwToFloat8RowwiseBwDynamic.apply(
        #     outputs,
        #     mod.linear_mm_config,
        #     mod.config.cast_config_grad_output.target_dtype,
        #     -1,
        # )
        # outputs = NoopFwToFloat8BwDynamic.apply(
        #     outputs,
        #     mod.linear_mm_config,
        #     mod.config.cast_config_grad_output.target_dtype,
        # )

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


class PrepareFloat8ModuleInput(PrepareModuleInput):
    """
    Like `PrepareModuleInput`, but with all-gather in float8. This
    currently assumes tensorwise scaling.

    The only difference from `PrepareModuleInput` is that
    after we prepare the input DTensor, we cast the input to DTensor(Float8Tensor)
    This is to ensure the float8 cast happens before the all-gather (i.e. Shard -> Replicate)
    so that if there are multiple float8 users of the input activation, we perform fp8 allgather
    only once.
    FP8 Args:
      float8_dtype (torch.dtype, optional): control what float8 dtype to cast to when prepare the module input,
          we currently only support torch.float8_e4m3fn. default: torch.float8_e4m3fn
      fwd_config_submodule_fqn (str, optional): the fqn of the submodule that contains the forward config used
          for the float8 cast. If not specified, we will search for the Float8Linear in the submodules
          and use the forward config from that module, in this case all module's forward config must be
          the same.
    """

    def __init__(
        self,
        *,
        input_layouts=None,
        desired_input_layouts=None,
        input_kwarg_layouts=None,
        desired_input_kwarg_layouts=None,
        use_local_output=False,
        float8_dtype=torch.float8_e4m3fn,
        fwd_config_submodule_fqn=None,
    ):
        super().__init__(
            input_layouts=input_layouts,
            desired_input_layouts=desired_input_layouts,
            input_kwarg_layouts=input_kwarg_layouts,
            desired_input_kwarg_layouts=desired_input_kwarg_layouts,
            use_local_output=use_local_output,
        )

        # fp8 specific fields
        self.float8_dtype = float8_dtype
        self.linear_mm_config = None
        self.fwd_config_submodule_fqn = fwd_config_submodule_fqn

        if self.float8_dtype != torch.float8_e4m3fn:
            raise NotImplementedError(
                "PrepareFloat8ModuleInput only support casting to float8_e4m3fn for now"
            )

    def _prepare_input_arg(self, input, mesh, input_layout, desired_layout):
        if input_layout is not None:
            if isinstance(input, DTensor):
                # TODO: re-enable the check once we fix the compile path
                # assert inp.placements[0] == input_layout
                dt_inp = input
            else:
                assert isinstance(input, torch.Tensor), (
                    "expecting input to be a torch.Tensor!"
                )
                dt_inp = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )

            dt_inp = hp_tensor_to_float8_dynamic(
                dt_inp,
                e4m3_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )  # DTensor(Float8Tensor)
            if desired_layout is not None and input_layout != desired_layout:
                dt_inp = dt_inp.redistribute(placements=(desired_layout,))

            return dt_inp.to_local() if self.use_local_output else dt_inp
        else:
            return input

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        from torchao.float8.float8_linear import Float8Linear

        if self.fwd_config_submodule_fqn is not None:
            fwd_linear = module.get_submodule(self.fwd_config_submodule_fqn)
            assert isinstance(fwd_linear, Float8Linear)
            self.linear_mm_config = fwd_linear.linear_mm_config
        else:
            # search for ScaledMM configs for all the submodules and make sure they are the same
            for mod in module.modules():
                if isinstance(mod, Float8Linear):
                    if self.linear_mm_config is None:
                        self.linear_mm_config = mod.linear_mm_config
                    else:
                        assert self.linear_mm_config == mod.linear_mm_config, (
                            "All the Float8Linear modules should have same linear_mm_config!"
                        )

        assert self.linear_mm_config is not None
        super()._apply(module, device_mesh)
        return module


@torch._dynamo.allow_in_graph
class NoopFwToFloat8RowwiseBwDynamic(torch.autograd.Function):
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
        fp8_tensor = hp_tensor_to_float8_dynamic(
            gradY,
            ctx.target_dtype,
            ctx.linear_mm_config,
            gemm_input_role=GemmInputRole.GRAD_OUTPUT,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=ctx.axiswise_dim,
        )
        return fp8_tensor, None, None, None



def is_row_major(tensor: torch.Tensor) -> bool:
    assert tensor.dim() >= 2
    return tensor.stride(-2) > tensor.stride(-1) and tensor.stride(-1) == 1

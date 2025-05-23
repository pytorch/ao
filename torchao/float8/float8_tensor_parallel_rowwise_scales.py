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
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)

from torchao.float8.config import ScalingGranularity, ScalingType, e4m3_dtype, Float8LinearConfig
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    NoopFwToFloat8BwDynamic,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import GemmInputRole, LinearMMConfig, Float8Tensor

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


@torch._dynamo.allow_in_graph
class matmul_with_fp8_input_and_input_t(torch.autograd.Function):
    """
    Differentiable scaled mm between input and weight tensor, with the 
    input tensor already given in float8 row-major format (for forward) 
    and float8 column-major format (for backward). The weight tensor 
    can be high precision or float8.
    """

    @staticmethod
    def forward(
        ctx,
        input: Float8Tensor,
        input_t: Float8Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_t, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if tensor_already_casted_to_fp8(input):
            input_maybe_fp8 = input
        elif c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input,
                c.cast_config_input.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_input.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(weight_hp_t):
            weight_maybe_fp8_t = weight_hp_t
        elif c.cast_config_weight.scaling_type is ScalingType.DISABLED:
            weight_maybe_fp8_t = weight_hp_t
        else:
            weight_maybe_fp8_t = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight.target_dtype,
                linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )
        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        if input_t.shape[-1] != input.shape[-2]:
            torch.distributed.breakpoint()
        print(f"fwd input: {input.shape}, input_t: {input_t.shape}, weight_hp_t: {weight_hp_t.shape}")
        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_t, weight_hp_t = ctx.saved_tensors
        c = ctx.config

        print(f"bwd grad_output: {grad_output.shape}, input_t: {input_t.shape}, weight_hp_t: {weight_hp_t.shape}")
        # the reshapes are needed in order to make the shapes compatible with
        # torch.mm
        grad_output_orig_shape = grad_output.shape
        grad_output_reshaped = grad_output.reshape(-1, grad_output_orig_shape[-1])

        #
        # calculate grad_input
        #

        if tensor_already_casted_to_fp8(grad_output_reshaped):
            # TODO(future PR): this var name is axiswise-specific, fix it
            grad_output_reshaped_maybe_fp8_dim0 = grad_output_reshaped
        elif c.cast_config_grad_output.scaling_type is ScalingType.DISABLED:
            grad_output_reshaped_maybe_fp8_dim0 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_grad_output.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(weight_hp_t):
            # TODO(future PR): var name is axiswise specific, fix it
            weight_t_maybe_fp8 = weight_hp_t
        elif c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8 = weight_hp_t
        else:
            if (
                c.cast_config_weight_for_grad_input.scaling_granularity
                is ScalingGranularity.AXISWISE
            ):
                # workaround from https://github.com/pytorch/pytorch/issues/141881
                # to avoid saving float8 weight from forward to backward when
                # FSDP is on: add a fake dependency on `grad_output`.
                g_reshaped = grad_output.reshape(-1, grad_output.shape[-1]) * 0
                zero = g_reshaped[:1] * 0
                weight_hp_t = weight_hp_t + zero

            # Note: we need https://github.com/pytorch/pytorch/issues/136267
            # to be solved to have a chance to reuse max(abs(weight, dim=...))
            # from the forward to get max(abs(weight)) here without reading
            # the entire tensor.
            weight_t_maybe_fp8 = hp_tensor_to_float8_dynamic(
                weight_hp_t,
                c.cast_config_weight_for_grad_input.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.WEIGHT,
                scaling_granularity=c.cast_config_weight_for_grad_input.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_weight_for_grad_input.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        grad_input = torch.mm(
            weight_t_maybe_fp8,
            grad_output_reshaped_maybe_fp8_dim0,
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        #
        # calculate grad_weight
        #
        grad_output_t = grad_output.transpose(-2, -1)
        grad_output_t_reshaped = grad_output_t.reshape(-1, grad_output_t.shape[-1])
        if tensor_already_casted_to_fp8(grad_output_t_reshaped):
            # TODO(future PR): var name is axiswise specific, fix it
            grad_output_t_maybe_fp8 = grad_output_t_reshaped
        elif (
            c.cast_config_grad_output_for_grad_weight.scaling_type
            is ScalingType.DISABLED
        ):
            grad_output_t_maybe_fp8 = grad_output_t_reshaped
        else:
            grad_output_t_maybe_fp8 = hp_tensor_to_float8_dynamic(
                grad_output_t_reshaped,
                c.cast_config_grad_output_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_grad_output_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(input_t):
            # TODO(future PR): var name is axiswise specific, fix it
            input_t_maybe_fp8 = input_t
        elif c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_t_maybe_fp8 = input_t
        else:
            input_t_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_t,
                c.cast_config_input_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    -1, c.cast_config_input_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        grad_weight = torch.mm(
            input_t_maybe_fp8,
            grad_output_t_maybe_fp8,
        )

        # the 2nd input (col-major) is created only for fp8 rowwise all-gather, it should have no grad.
        # we cannot pass "None" because under the hood, pytorch will create a regular, non-DTensor filled
        # with 0s, which is incompatible for subsequent ops with other DTensors during backward.
        # So we have to manually create a DTensor with 0s here.
        empty_grad = grad_input.clone().reshape(input_t_maybe_fp8.shape)
        empty_grad.fill_(0.0)
        return grad_input, empty_grad, grad_weight.t(), None, None

class Float8ColwiseParallel(ColwiseParallel):
    """
    Like `ColwiseParallel`, but with all-gather in float8 with rowwise scales.
    """

    @staticmethod
    def _prepare_input_fn(
        input_layouts, desired_input_layouts, mod, inputs, device_mesh,
    ):
        # annotate module input placements/sharding with input_layouts
        if len(inputs) == 1:
            input = inputs[0]
            input_t = inputs[0].transpose(-2, -1).contiguous()
        elif len(inputs) == 2:
            input, input_t = inputs[0], inputs[1]
        else:
            raise ValueError("expected inputs to be length 1 or 2, but got", len(inputs))

        # convert input to DTensor
        if isinstance(input, DTensor):
            dt_input = input
        else:
            dt_input = DTensor.from_local(
                input, device_mesh, input_layouts, run_check=False
            )

        # convert input to fp8 rowwise
        if not tensor_already_casted_to_fp8(dt_input):
            dt_input_fp8 = hp_tensor_to_float8_dynamic(
                dt_input,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)
        else:
            dt_input_fp8 = dt_input

        # convert input_t to DTensor
        if isinstance(input_t, DTensor):
            dt_input_t = input_t
        else:
            dt_input_t = DTensor.from_local(
                input_t, device_mesh, input_layouts, run_check=False
            )
        # convert input_t to fp8 rowwise
        if not tensor_already_casted_to_fp8(input_t):
            dt_input_t_fp8 = hp_tensor_to_float8_dynamic(
                dt_input_t,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)
        else:
            dt_input_t_fp8 = input_t

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            if dt_input_fp8 is not None:
                dt_input_fp8 = dt_input_fp8.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            if dt_input_t_fp8 is not None:
                dt_input_t_fp8 = dt_input_t_fp8.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
        return dt_input_fp8, dt_input_t_fp8

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
        if len(inputs) == 1:
            input = inputs[0]
            input_t = inputs[0].transpose(-2, -1).contiguous()
        elif len(inputs) == 2:
            input, input_t = inputs[0], inputs[1]
        else:
            raise ValueError("expected inputs to be length 1 or 2, but got", len(inputs))

        # convert input to DTensor
        if isinstance(input, DTensor):
            dt_input = input
        else:
            dt_input = DTensor.from_local(
                input, device_mesh, input_layouts, run_check=False
            )

        # convert input to fp8 rowwise
        if not tensor_already_casted_to_fp8(dt_input):
            dt_input_fp8 = hp_tensor_to_float8_dynamic(
                dt_input,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)
        else:
            dt_input_fp8 = dt_input

        # convert input_t to DTensor
        if isinstance(input_t, DTensor):
            dt_input_t = input_t
        else:
            # danvm: sharding dim for transposed input is wrong here, since we have transposed dims
            dt_input_t = DTensor.from_local(
                input_t, device_mesh, input_layouts, run_check=False
            )
        # convert input_t to fp8 rowwise
        if not tensor_already_casted_to_fp8(input_t):
            dt_input_t_fp8 = hp_tensor_to_float8_dynamic(
                dt_input_t,
                mod.config.cast_config_input.target_dtype,
                mod.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)
        else:
            dt_input_t_fp8 = input_t

        # transform the input layouts to the desired layouts of ColwiseParallel
        if input_layouts != desired_input_layouts:
            if dt_input_fp8 is not None:
                dt_input_fp8 = dt_input_fp8.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
            if dt_input_t_fp8 is not None:
                dt_input_t_fp8 = dt_input_t_fp8.redistribute(
                    placements=desired_input_layouts, async_op=True
                )
        return dt_input_fp8, dt_input_t_fp8

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


class PrepareFloat8ModuleInput(PrepareModuleInput):
    """
    Like `PrepareModuleInput`, but with all-gather in float8 with row-wise scales.

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
                dt_input = input
                dt_input_t = input.transpose(-2, -1).contiguous()
            else:
                assert isinstance(input, torch.Tensor), (
                    "expecting input to be a torch.Tensor!"
                )
                dt_input = DTensor.from_local(
                    input, mesh, (input_layout,), run_check=False
                )
                dt_input_t = DTensor.from_local(
                    input.transpose(-2, -1).contiguous(),
                    mesh, 
                    (input_layout,), 
                    run_check=False,
                )
            
            
            dt_input_fp8 = hp_tensor_to_float8_dynamic(
                dt_input,
                e4m3_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)

            dt_input_t_fp8 = hp_tensor_to_float8_dynamic(
                dt_input_t,
                e4m3_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=ScalingGranularity.AXISWISE,
                axiswise_dim=-1,
            )  # DTensor(Float8Tensor)

            if desired_layout is not None and input_layout != desired_layout:
                dt_input_fp8 = dt_input_fp8.redistribute(placements=(desired_layout,))
                dt_input_t_fp8 = dt_input_t_fp8.redistribute(placements=(desired_layout,))

            dt_input_fp8 = dt_input_fp8.to_local() if self.use_local_output else dt_input_fp8
            dt_input_t_fp8 = dt_input_t_fp8.to_local() if self.use_local_output else dt_input_t_fp8
            return dt_input_fp8, dt_input_t_fp8
        else:
            # for non-DTensor input (e.g. freqs_cis buffer in RoPE) we don't need to do anything.
            return input, None

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

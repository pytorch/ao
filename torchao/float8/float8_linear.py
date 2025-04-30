# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear`.
"""

from typing import Optional

import torch
import torch.utils.checkpoint as checkpoint
from torch.distributed._tensor import DTensor

from torchao.float8.config import Float8LinearConfig, ScalingGranularity, ScalingType
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_scaling_utils import (
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_dynamic,
)
from torchao.float8.float8_tensor import (
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import tensor_to_scale
from torchao.float8.fsdp_utils import WeightWithDynamicFloat8CastTensor
from torchao.float8.float8_tensor import Float8Tensor


def _get_weight_scale(
    weight: torch.Tensor,
    scaling_type_weight: ScalingType,
    config: Float8LinearConfig,
) -> Optional[torch.Tensor]:
    if tensor_already_casted_to_fp8(weight):
        return None
    assert scaling_type_weight is ScalingType.DYNAMIC
    return tensor_to_scale(weight, config.cast_config_weight.target_dtype)


def _cast_weight_to_float8_t(
    weight: torch.Tensor,
    config: Float8LinearConfig,
    linear_mm_config: LinearMMConfig,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if tensor_already_casted_to_fp8(weight):
        return weight.t()
    weight_fp8 = hp_tensor_and_scale_to_float8(
        weight,
        weight_scale,
        config.cast_config_weight.target_dtype,
        linear_mm_config,
        gemm_input_role=GemmInputRole.WEIGHT,
    )
    return weight_fp8.t()

@torch._dynamo.allow_in_graph
class matmul_with_hp_or_float8_args(torch.autograd.Function):
    """
    Like torch.matmul, but with the arguments in either high precision or float8.
    * if the arguments are in high precision, they are cast to float8 according
      to the specified config
    * if the arguments are in float8, we assume the cast honored the config
    """

    @staticmethod
    def forward(
        ctx,
        input_hp: torch.Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        ctx.save_for_backward(input_hp, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if tensor_already_casted_to_fp8(input_hp):
            input_maybe_fp8 = input_hp
        elif c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input_hp
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_hp,
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
        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_hp, weight_hp_t = ctx.saved_tensors
        c = ctx.config

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
                    -1, c.cast_config_grad_output.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(weight_hp_t):
            # TODO(future PR): var name is axiswise specific, fix it
            weight_t_maybe_fp8_dim0 = weight_hp_t
        elif c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8_dim0 = weight_hp_t
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
            weight_t_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
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
            grad_output_reshaped_maybe_fp8_dim0,
            weight_t_maybe_fp8_dim0.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        input_hp_orig_shape = input_hp.shape
        input_hp_reshaped = input_hp.reshape(-1, input_hp_orig_shape[-1])

        #
        # calculate grad_weight
        #

        if tensor_already_casted_to_fp8(grad_output_reshaped):
            # TODO(future PR): var name is axiswise specific, fix it
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        elif (
            c.cast_config_grad_output_for_grad_weight.scaling_type
            is ScalingType.DISABLED
        ):
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_grad_output_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(input_hp_reshaped):
            # TODO(future PR): var name is axiswise specific, fix it
            input_reshaped_maybe_fp8_dim1 = input_hp_reshaped
        elif c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_reshaped_maybe_fp8_dim1 = input_hp_reshaped
        else:
            input_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                input_hp_reshaped,
                c.cast_config_input_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_input_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        grad_weight = torch.mm(
            grad_output_reshaped_maybe_fp8_dim1.t(),
            input_reshaped_maybe_fp8_dim1,
        )

        empty_grads = None, None

        return grad_input, grad_weight.t(), *empty_grads


@torch._dynamo.allow_in_graph
class matmul_with_fp8_input_row_and_col_major(torch.autograd.Function):
    """
    Differentiable scaled mm between input and weight tensor, with the 
    input tensor already given in float8 row-major format (for forward) 
    and float8 column-major format (for backward). The weight tensor 
    can be high precision or float8.
    """

    @staticmethod
    def forward(
        ctx,
        input_row_major: Float8Tensor,
        input_col_major: Float8Tensor,
        weight_hp_t: torch.Tensor,
        linear_mm_config: LinearMMConfig,
        config: Float8LinearConfig,
    ):
        assert input_col_major.dim() == 2, "input_col_major must be 2D Float8Tensor"
        assert input_col_major.to_local()._axiswise_dim is not None, "input_col_major must be axiswise"

        ctx.save_for_backward(input_col_major, weight_hp_t)
        ctx.linear_mm_config = linear_mm_config
        ctx.config = config

        c = config

        if tensor_already_casted_to_fp8(input_row_major):
            input_maybe_fp8 = input_row_major
        elif c.cast_config_input.scaling_type is ScalingType.DISABLED:
            input_maybe_fp8 = input_row_major
        else:
            input_maybe_fp8 = hp_tensor_to_float8_dynamic(
                input_row_major,
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
        orig_shape = input_maybe_fp8.shape
        input_maybe_fp8_reshaped = input_maybe_fp8.reshape(-1, orig_shape[-1])
        res_bits = torch.mm(input_maybe_fp8_reshaped, weight_maybe_fp8_t)
        res_bits = res_bits.reshape(*orig_shape[:-1], res_bits.shape[-1])
        return res_bits

    @staticmethod
    def backward(ctx, grad_output):
        input_fp8_col_major, weight_hp_t = ctx.saved_tensors
        c = ctx.config
        assert input_fp8_col_major.to_local()._axiswise_dim is not None, "input_col_major must be axiswise"

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
                    -1, c.cast_config_grad_output.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(weight_hp_t):
            # TODO(future PR): var name is axiswise specific, fix it
            weight_t_maybe_fp8_dim0 = weight_hp_t
        elif c.cast_config_weight_for_grad_input.scaling_type is ScalingType.DISABLED:
            weight_t_maybe_fp8_dim0 = weight_hp_t
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
            weight_t_maybe_fp8_dim0 = hp_tensor_to_float8_dynamic(
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
            grad_output_reshaped_maybe_fp8_dim0,
            weight_t_maybe_fp8_dim0.t(),
        )
        grad_input = grad_input.reshape(
            *grad_output_orig_shape[:-1], grad_input.shape[-1]
        )

        #
        # calculate grad_weight
        #

        if tensor_already_casted_to_fp8(grad_output_reshaped):
            # TODO(future PR): var name is axiswise specific, fix it
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        elif (
            c.cast_config_grad_output_for_grad_weight.scaling_type
            is ScalingType.DISABLED
        ):
            grad_output_reshaped_maybe_fp8_dim1 = grad_output_reshaped
        else:
            grad_output_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                grad_output_reshaped,
                c.cast_config_grad_output_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.GRAD_OUTPUT,
                scaling_granularity=c.cast_config_grad_output_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_grad_output_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        if tensor_already_casted_to_fp8(input_fp8_col_major):
            # TODO(future PR): var name is axiswise specific, fix it
            input_reshaped_maybe_fp8_dim1 = input_fp8_col_major
        elif c.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED:
            input_reshaped_maybe_fp8_dim1 = input_fp8_col_major
        else:
            input_reshaped_maybe_fp8_dim1 = hp_tensor_to_float8_dynamic(
                input_fp8_col_major,
                c.cast_config_input_for_grad_weight.target_dtype,
                ctx.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
                scaling_granularity=c.cast_config_input_for_grad_weight.scaling_granularity,
                axiswise_dim=get_maybe_axiswise_dim(
                    0, c.cast_config_input_for_grad_weight.scaling_granularity
                ),
                round_scales_to_power_of_2=c.round_scales_to_power_of_2,
            )

        grad_weight = torch.mm(
            grad_output_reshaped_maybe_fp8_dim1.t(),
            input_reshaped_maybe_fp8_dim1,
        )
        return grad_input, grad_input.reshape(input_reshaped_maybe_fp8_dim1.shape), grad_weight.t(), None, None


class Float8Linear(torch.nn.Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp8 compute.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8LinearConfig
        """

        config = kwargs.pop("config")
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of input, weight, grad_output
        self.scaling_type_input = config.cast_config_input.scaling_type
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        self.scaling_type_grad_output = config.cast_config_grad_output.scaling_type
        self.config = config

        self.linear_mm_config = LinearMMConfig(
            # output
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_output.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_input
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_input.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
            # grad_weight
            ScaledMMConfig(
                config.emulate,
                self.config.gemm_config_grad_weight.use_fast_accum,
                False,
                self.config.pad_inner_dim,
            ),
        )

    def forward(self, input_row_major: torch.Tensor, input_col_major: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input_row_major = input_row_major.to(autocast_dtype)
            if input_col_major is not None:
                input_col_major = input_col_major.to(autocast_dtype)

        has_any_axiswise_scaling = any(
            cc.scaling_granularity is ScalingGranularity.AXISWISE
            for cc in [
                self.config.cast_config_input,
                self.config.cast_config_weight,
                self.config.cast_config_grad_output,
                self.config.cast_config_input_for_grad_weight,
                self.config.cast_config_weight_for_grad_input,
                self.config.cast_config_grad_output_for_grad_weight,
            ]
        )

        weight_maybe_fp8_t = self.weight.t()

        # TODO(future PR): check for axiswise scaling for input, weight,
        # grad_output separately instead of together
        if not has_any_axiswise_scaling:
            # If force_recompute_fp8_weight_in_bwd, we only recompute the fp8 weight,
            # weight_scale should be saved.
            weight_scale = _get_weight_scale(
                self.weight, self.scaling_type_weight, self.config
            )

            if self.config.force_recompute_fp8_weight_in_bwd:
                weight_fp8_t = checkpoint.checkpoint(
                    _cast_weight_to_float8_t,
                    self.weight,
                    self.config,
                    self.linear_mm_config,
                    weight_scale,
                )
            else:
                weight_fp8_t = _cast_weight_to_float8_t(
                    self.weight,
                    self.config,
                    self.linear_mm_config,
                    weight_scale,
                )

            weight_maybe_fp8_t = weight_fp8_t

        fp8_rowwise_all_gather = (
           input_row_major is not None and 
           input_col_major is not None and
           isinstance(input_row_major, DTensor) and
           isinstance(input_col_major, DTensor) and
           isinstance(input_row_major.to_local(), Float8Tensor) and
           isinstance(input_col_major.to_local(), Float8Tensor)
        )
        if fp8_rowwise_all_gather:
            output = matmul_with_fp8_input_row_and_col_major.apply(
                input_row_major,
                input_col_major,
                weight_maybe_fp8_t,
                self.linear_mm_config,
                self.config,
            )
        else:
            output = matmul_with_hp_or_float8_args.apply(
                input_row_major,
                weight_maybe_fp8_t,
                self.linear_mm_config,
                self.config,
            )
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    def extra_repr(self):
        c = self.config
        ci = f"i:{c.cast_config_input.short_str()}"
        cw = f"w:{c.cast_config_weight.short_str()}"
        cgo = f"go:{c.cast_config_grad_output.short_str()}"
        parts = [ci, cw, cgo]
        if c.cast_config_input_for_grad_weight != c.cast_config_input:
            parts.append(f"i_gw:{c.cast_config_input_for_grad_weight.short_str()}")
        if c.cast_config_weight_for_grad_input != c.cast_config_weight:
            parts.append(f"w_gi:{c.cast_config_weight_for_grad_input.short_str()}")
        if c.cast_config_grad_output_for_grad_weight != c.cast_config_grad_output:
            parts.append(
                f"go_gw:{c.cast_config_grad_output_for_grad_weight.short_str()}"
            )
        cast_config_str = ",".join(parts)
        s = f'{super().extra_repr()}, cast_configs={cast_config_str}"'
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8LinearConfig] = None,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8LinearConfig()
        with torch.device("meta"):
            new_mod = cls(
                mod.in_features,
                mod.out_features,
                bias=False,
                config=config,
            )
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        # TODO(future PR): see if we can simplify ^ now that delayed scaling is deleted
        if config.enable_fsdp_float8_all_gather:
            assert config.cast_config_weight.scaling_type is ScalingType.DYNAMIC
            new_mod.weight = torch.nn.Parameter(
                WeightWithDynamicFloat8CastTensor(
                    new_mod.weight,
                    new_mod.linear_mm_config,
                    new_mod.config.cast_config_weight.target_dtype,
                ),
                requires_grad=new_mod.weight.requires_grad,
            )

        return new_mod

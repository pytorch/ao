# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import enum
from dataclasses import dataclass
from typing import Optional

import torch


class ScalingType(enum.Enum):
    DELAYED = "delayed"
    DYNAMIC = "dynamic"
    STATIC = "static"
    # ScalingType.DISABLED means "skip scaling for this tensor, leave it in
    # its original precision.
    DISABLED = "disabled"

    def short_str(self):
        if self is ScalingType.DELAYED:
            return "del"
        elif self is ScalingType.DYNAMIC:
            return "dyn"
        elif self is ScalingType.STATIC:
            return "sta"
        else:
            assert self is ScalingType.DISABLED
            return "dis"


class ScalingGranularity(enum.Enum):
    """
    Defines the granularity of scaling strategies for casting to float8
    """

    # A single scaling factor for the entire tensor
    TENSORWISE = "tensorwise"
    # Scaling factors computed along one axis of the tensor, reducing it to
    # size 1.
    AXISWISE = "axiswise"

    def short_str(self):
        if self is ScalingGranularity.TENSORWISE:
            return "ten"
        else:
            assert self is ScalingGranularity.AXISWISE
            return "axs"


@dataclass(frozen=True)
class CastConfig:
    """
    Configuration for maybe casting a single tensor to float8
    """

    scaling_type: ScalingType = ScalingType.DYNAMIC
    scaling_granularity: ScalingGranularity = ScalingGranularity.TENSORWISE
    static_scale: Optional[torch.Tensor] = None

    def short_str(self):
        return f"{self.scaling_type.short_str()}_{self.scaling_granularity.short_str()}"

    def __post_init__(self):
        if self.scaling_type is ScalingType.STATIC:
            assert (
                self.static_scale is not None
            ), "static_scale must be specified for static scaling"
        if self.scaling_granularity is ScalingGranularity.AXISWISE:
            assert self.scaling_type is ScalingType.DYNAMIC, \
                "only dynamic scaling type is supported for axiswise scaling granularity"

@dataclass(frozen=True)
class DelayedScalingConfig:
    """
    Configuration for delayed scaling.

    Note: for now, `history_len` values must be the same for all layers in the
    model using delayed scaling.

    TODO(future): serialization for recipes
    """

    # Controls the history length of amax buffers
    history_len: int = 16

    # Controls the way to calculate current scale from amax history
    # TODO(future): add other functions as needed, hardcoded or user defined
    scale_fn_name: str = "max"

    def __post_init__(self):
        assert (
            self.scale_fn_name == "max"
        ), f"{self.scale_fn_name} is not implemented yet. Only max is supported for now."


@dataclass(frozen=True)
class Float8GemmConfig:
    """
    Configuration for a float8 gemm.
    """

    # If True, fast accumulation in lower precision is used.
    # Note: this flag is currently a no-op if emulation is turned on.
    use_fast_accum: bool = False


@dataclass(frozen=True)
class Float8LinearConfig:
    """
    Configuration for converting a `torch.nn.Linear` module to float8
    for training.
    """

    #
    # Per-tensor configuration for casting of `input`, `weight`, `grad_output`
    # for the operands of gemms calculating `output`, `grad_weight`, and `grad_input`.
    #
    # Note: 
    # 1. if `cast_config_input_for_grad_weight` is None, then 
    #    `cast_config_input` is used for scaling `input` for both gemms that
    #    use `input.  
    # 2. if `cast_config_input_for_grad_weight` is specified, then 
    #    a. `cast_config_input` is used for scaling `input` for the gemm that calculates
    #       `output`
    #    b. `cast_config_input_for_grad_weight` is used for scaling `input` for
    #       the gemm that calculates `grad_weight`
    # 3. the same behavior holds for `cast_config_weight` and `cast_config_grad_output`.
    #
    # `input`
    cast_config_input: CastConfig = CastConfig()
    cast_config_input_for_grad_weight: Optional[CastConfig] = None
    # `weight`
    cast_config_weight: CastConfig = CastConfig()
    cast_config_weight_for_grad_input: Optional[CastConfig] = None
    # `grad_output`
    cast_config_grad_output: CastConfig = CastConfig()
    cast_config_grad_output_for_grad_weight: Optional[CastConfig] = None

    #
    # Per-gemm configuration for gemms calculating `output`, `grad_input` and
    # `grad_weight`
    # TODO(this PR): throw warning if fast_accum False is used with axiswise scaling
    #
    gemm_config_output: Float8GemmConfig = Float8GemmConfig(use_fast_accum=True)
    gemm_config_grad_input: Float8GemmConfig = Float8GemmConfig()
    gemm_config_grad_weight: Float8GemmConfig = Float8GemmConfig()

    #
    # Per-linear configuration
    #

    # If True, on the first iteration of Float8Linear the amaxes will be
    # initialized with the incoming data. As of 2023-12-30, this doesn't work
    # with autocast + torch.compile + FSDP. Enabling this option is nice for
    # testing, but this is not necessary for real training jobs.
    enable_amax_init: bool = True

    # If True, pre-forward and post-forward functions are run. As of 2023-12-30,
    # this doesn't work with autocast + torch.compile + FSDP. Enabling this
    # option is useful for safety, but not strictly necessary.
    enable_pre_and_post_forward: bool = True

    # If True, then uses a tensor subclass for the float8 linear module's weight that
    # implements pre/post-all-gather methods to do float8 all-gather with FSDP2.
    enable_fsdp_float8_all_gather: bool = False

    # If True, then prior to performing the fp8 scaled mamtmul we will pad the
    # inner dimension of a (dim 1) and b (dim 2) with 0s. This is needed for matmuls
    # _scaled_mm since it has the strong constraint that for M,N,K  N, K must be a multiple of 16.
    # This can cause a memory spike however so we keep this off by default.
    pad_inner_dim: bool = False

    # If True, emulation is used instead of hardware accelerated gemm
    emulate: bool = False

    # Configuration for delayed scaling
    # Note: this is actually applied per-tensor, but only using the same
    # configuration for all tensors and layers in the model is currently
    # supported. If in the future we add support for a more fine grained
    # configuration, this field may move to per-tensor configs.
    delayed_scaling_config: DelayedScalingConfig = DelayedScalingConfig()

    # If the option is enabled, fp8_weight will always be re-computed in backward.
    # It's recommended to enable this flag when using FSDP.
    # Otherwise, the entire fp8_weight, instead of the sharded weight may be saved.
    # If using outer activation checkpointing context or SAC, you may disable this option
    # and handle the recomputation of fp8 weight in your customized AC context.
    #
    # Details:
    # When using float8 training with FSDP, the original weight is sharded; fp8_weight (in forward) and fp8_weight_transpose (in backward) are used by the model.
    # However, when partitioning the forward_backward graph, torch.compile may decide to
    # save the fp8_weight_transpose for backward, which is an un-sahrded weight and costs a high memory utilization.
    # The longer-term solution is to let compile decide how to partition the graph with optimal computation and memory savings.
    # For now, we use the checkpointing api to force the recomputation of fp8 weight in backward.
    # TODO(future PR): either enable by default or have a warning and set up the
    # tests so that the warning does not spam the CI stdout.

    force_recompute_fp8_weight_in_bwd: bool = False

    def __post_init__(self):
        # Populate the additional cast overrides, if the user did not specify them
        # Note: this hacks around the frozen-ness of this dataclass
        # by using `object.__setattr__`.  This is fine, as what we really need
        # is for this object to be frozen after `__post_init__` for torch.compile
        # to work.
        # Source of hack: https://stackoverflow.com/a/65959419/
        if self.cast_config_input_for_grad_weight is None:
            object.__setattr__(self, "cast_config_input_for_grad_weight", self.cast_config_input)
        if self.cast_config_weight_for_grad_input is None:
            object.__setattr__(self, "cast_config_weight_for_grad_input", self.cast_config_weight)
        if self.cast_config_grad_output_for_grad_weight is None:
            object.__setattr__(self, "cast_config_grad_output_for_grad_weight", self.cast_config_grad_output)

        # float8 all-gather only supports tensorwise, in the future may support blockwise
        if self.cast_config_weight.scaling_granularity != ScalingGranularity.TENSORWISE:
            assert not self.enable_fsdp_float8_all_gather, \
                f"enable_fsdp_float8_all_gather only supports tensorwise scaling granularity, got {self.cast_config_weight.scaling_granularity}"

        # save some characters in the compatibility checks below
        cc_i = self.cast_config_input
        cc_w = self.cast_config_weight
        cc_go = self.cast_config_grad_output
        cc_i_gw = self.cast_config_input_for_grad_weight
        cc_w_gi = self.cast_config_weight_for_grad_input
        cc_go_gw = self.cast_config_grad_output_for_grad_weight
        # for now, we only have gemm kernels where both operands are either both
        # in high precision, or both in float8. In the future, this may be relaxed.
        # TODO(future): make the float8 check more precise with the specific dtypes.
        for cc1, cc2, gemm_name in (
            (cc_i, cc_w, "output"),
            (cc_go, cc_w_gi, "grad_input"),
            (cc_i_gw, cc_go_gw, "grad_weight"),
        ):
            is_disabled_1 = cc1.scaling_type is ScalingType.DISABLED
            is_disabled_2 = cc1.scaling_type is ScalingType.DISABLED
            assert is_disabled_1 == is_disabled_2, \
                f"incompatible operand precision for {gemm_name}"


# If True, use 'fnuz' float8 types for calculations.
# Currently, ROCm only supports fnuz variants.
# TODO(future PR): move this to Float8LinearConfig
use_fnuz_dtype = False


# Pre-made recipes for common configurations
# TODO(future PR): go through a round of design on this, and eventually expose
# as a top level public API.
class Float8LinearRecipeName(enum.Enum):
    ALL_TENSORWISE = "all_tensorwise"
    ALL_AXISWISE = "all_axiswise"
    LW_AXISWISE_WITH_GW_HP = "lw_axiswise_with_gw_hp"


def recipe_name_to_linear_config(
    recipe_name: Float8LinearRecipeName,
) -> Float8LinearConfig:
    """
    Input: `Float8LinearRecipeName` value
    Output: a `Float8LinearConfig` configured to implement the recipe
    """

    if recipe_name is Float8LinearRecipeName.ALL_TENSORWISE:
        # Default, dynamic per-tensor scaling with the cuBLAS tensorwise kernel
        return Float8LinearConfig()

    elif recipe_name is Float8LinearRecipeName.ALL_AXISWISE:
        # dynamic axiswise scaling with the CUTLASS rowwise kernel
        cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
        cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
        cc_go = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
        
        # The current rowwise CUTLASS kernels in `torch._scaled_mm` are only
        # fast with `use_fast_accum=True`. Note that rowwise scaling is more
        # accurate than tensorwise scaling, so the overall impact on accuracy
        # of tensorwise vs rowwise taking this flag into account will vary.
        gc_o = Float8GemmConfig(use_fast_accum=True)
        gc_gi = Float8GemmConfig(use_fast_accum=True)
        gc_gw = Float8GemmConfig(use_fast_accum=True)

        return Float8LinearConfig(
            cast_config_input=cc_i,
            cast_config_weight=cc_w,
            cast_config_grad_output=cc_go,
            gemm_config_output=gc_o,
            gemm_config_grad_input=gc_gi,
            gemm_config_grad_weight=gc_gw,
        )

    elif recipe_name is Float8LinearRecipeName.LW_AXISWISE_WITH_GW_HP:

        # lw's recipe for a modification on all-axiswise:
        #
        #   output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
        #   grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
        #   grad_weight_hp = input_t_hp @ grad_output_hp
        #
        # key characteristics:
        #   * increased accuracy for grad_weight
        #   * `input`, `weight` and `grad_output` now only need to be scaled 
        #     axiswise across a single dim compared to vanilla all-axiswise, 
        #     which is more amenable to fast kernels

        # output_hp = input_fp8_axiswise_dim0 @ weight_t_axiswise_dim1
        cc_i = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
        cc_w = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)

        # grad_input_hp = grad_output_fp8_axiswise_dim0 @ weight_fp8_tensorwise
        cc_go = CastConfig(scaling_granularity=ScalingGranularity.AXISWISE)
        cc_w_gi = CastConfig(scaling_granularity=ScalingGranularity.TENSORWISE)

        # grad_weight_hp = input_t_hp @ grad_output_hp
        cc_i_gw = CastConfig(scaling_type=ScalingType.DISABLED)
        cc_go_gw = CastConfig(scaling_type=ScalingType.DISABLED)

        # The current rowwise CUTLASS kernels in `torch._scaled_mm` are only
        # fast with `use_fast_accum=True`. Note that rowwise scaling is more
        # accurate than tensorwise scaling, so the overall impact on accuracy
        # of tensorwise vs rowwise taking this flag into account will vary.
        gc_o = Float8GemmConfig(use_fast_accum=True)
        gc_gi = Float8GemmConfig(use_fast_accum=True)
        gc_gw = Float8GemmConfig(use_fast_accum=True)

        return Float8LinearConfig(
            cast_config_input=cc_i,
            cast_config_weight=cc_w,
            cast_config_grad_output=cc_go,
            cast_config_input_for_grad_weight=cc_i_gw,
            cast_config_weight_for_grad_input=cc_w_gi,
            cast_config_grad_output_for_grad_weight=cc_go_gw,
            gemm_config_output=gc_o,
            gemm_config_grad_input=gc_gi,
            gemm_config_grad_weight=gc_gw,
        )

    else:
        raise AssertionError(f"unknown recipe_name {recipe_name}")

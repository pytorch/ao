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

    def short_str(self):
        if self is ScalingType.DELAYED:
            return "del"
        elif self is ScalingType.DYNAMIC:
            return "dyn"
        else:
            assert self is ScalingType.STATIC
            return "sta"


class ScalingGranularity(enum.Enum):
    """
    Defines the granularity of scaling strategies for casting to float8
    """

    # A single scaling factor for the entire tensor
    TENSORWISE = "tensorwise"
    # Scaling factors computed along one axis of the tensor, reducing it to
    # size 1.
    AXISWISE = "axiswise"


@dataclass(frozen=True)
class CastConfig:
    """
    Configuration for casting a single tensor to float8
    """

    scaling_type: ScalingType = ScalingType.DYNAMIC
    static_scale: Optional[torch.Tensor] = None

    def __post_init__(self):
        if self.scaling_type is ScalingType.STATIC:
            assert self.static_scale is not None, \
                "static_scale must be specified for static scaling"

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
    # Per-tensor configuration for `input`, `weight`, `grad_output`
    #
    cast_config_input: CastConfig = CastConfig()
    cast_config_weight: CastConfig = CastConfig()
    cast_config_grad_output: CastConfig = CastConfig()

    #
    # Per-gemm configuration for gemms calculating `output`, `grad_input` and
    # `grad_weight`
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


# If True, use 'fnuz' float8 types for calculations.
# Currently, ROCm only supports fnuz variants.
# TODO(future PR): move this to Float8LinearConfig
use_fnuz_dtype = False

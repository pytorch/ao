# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Stateful version of Float8Linear, created to keep Float8Linear simple and
only require code readers to read the stateful code if they care about delayed
or static scaling.
"""

from typing import Optional

import torch

from torchao.float8.config import Float8LinearConfig, ScalingGranularity, ScalingType
from torchao.float8.distributed_utils import tensor_already_casted_to_fp8
from torchao.float8.float8_linear import Float8Linear
from torchao.float8.float8_scaling_utils import (
    NoopFwToFloat8BwDelayed,
    NoopFwToFloat8BwDynamic,
    NoopFwToFloat8BwStatic,
    _maybe_initialize_amaxes_scales_for_float8_cast,
    get_maybe_axiswise_dim,
    hp_tensor_to_float8_delayed,
    hp_tensor_to_float8_dynamic,
    hp_tensor_to_float8_static,
)
from torchao.float8.float8_tensor import (
    Float8Tensor,
    GemmInputRole,
    LinearMMConfig,
    ScaledMMConfig,
    hp_tensor_and_scale_to_float8,
)
from torchao.float8.float8_utils import (
    tensor_to_amax,
    tensor_to_scale,
)
from torchao.float8.fsdp_utils import (
    WeightWithDelayedFloat8CastTensor,
    WeightWithDynamicFloat8CastTensor,
    WeightWithStaticFloat8CastTensor,
)


class StatefulFloat8Linear(Float8Linear):
    def __init__(self, *args, **kwargs):
        # Amax scales should always be kept as float32.
        self.always_float32_buffers = set()

        super().__init__(*args, **kwargs)

        # Convenience flag to skip code related to delayed scaling
        self.has_any_delayed_scaling = (
            self.scaling_type_input is ScalingType.DELAYED
            or self.scaling_type_weight is ScalingType.DELAYED
            or self.scaling_type_grad_output is ScalingType.DELAYED
        )

        self.create_buffers()

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = not self.config.enable_amax_init

        # pre_forward and post_forward are currently broken with FSDP
        # and torch.compile, this option can disable them
        # Note that when using `self.config.enable_pre_and_post_forward = False`,
        # it's recommended to also set `self.config.enable_amax_init = False`.
        # Otherwise, the amax buffer would never be marked as initialized and
        # would be initialized in every iteration.
        self.enable_pre_and_post_forward = self.config.enable_pre_and_post_forward

    def create_buffers(self):
        # Default values for history buffers, see above TODO
        history_len = self.config.delayed_scaling_config.history_len
        device = self.weight.device
        default_input = torch.finfo(self.config.cast_config_input.target_dtype).max
        default_weight = torch.finfo(self.config.cast_config_weight.target_dtype).max
        default_grad_output = torch.finfo(
            self.config.cast_config_grad_output.target_dtype
        ).max

        # Note: for now, create all the buffers if any are needed, to postpone
        # the work to make the scale and amax syncing and history calculation
        # handle a heterogeneous setup. We can do that work later if benchmarks
        # show it is worth doing.
        if self.has_any_delayed_scaling:
            self.register_always_float32_buffer(
                "fp8_amax_input", torch.tensor([default_input], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_input", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_input", torch.tensor([1.0], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_weight", torch.tensor([default_weight], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_weight", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_weight", torch.tensor([1.0], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_grad_output",
                torch.tensor([default_grad_output], device=device),
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_grad_output", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_grad_output", torch.tensor([1.0], device=device)
            )

        if self.config.cast_config_input.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_input",
                self.config.cast_config_input.static_scale.to(device),
            )
        if self.config.cast_config_weight.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_weight",
                self.config.cast_config_weight.static_scale.to(device),
            )
        if self.config.cast_config_grad_output.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_grad_output",
                self.config.cast_config_grad_output.static_scale.to(device),
            )

    def register_always_float32_buffer(
        self, name: str, tensor: Optional[torch.Tensor], persistent: bool = True
    ) -> None:
        self.register_buffer(name=name, tensor=tensor, persistent=persistent)
        self.always_float32_buffers.add(name)

    def _apply(self, fn, recurse=True):
        ret = super()._apply(fn, recurse)
        self.convert_amax_buffer_to_float32()
        return ret

    def convert_amax_buffer_to_float32(self):
        for key in self.always_float32_buffers:
            if self._buffers[key] is not None:
                self._buffers[key] = self._buffers[key].to(torch.float32)

    def cast_input_to_float8(self, input: torch.Tensor) -> torch.Tensor:
        is_amax_initialized = self.is_amax_initialized
        # Duplicate the autocast logic for F.linear, so that the output
        # of our module has the right original precision
        if torch.is_autocast_enabled():
            # For now, hardcode to GPU's autocast dtype
            # if we need CPU support in the future, we can add it
            autocast_dtype = torch.get_autocast_gpu_dtype()
            input = input.to(autocast_dtype)

        if self.scaling_type_input is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            _maybe_initialize_amaxes_scales_for_float8_cast(
                input,
                self.fp8_amax_input,
                self.fp8_amax_history_input,
                self.fp8_scale_input,
                scale_fn_name,
                self.config.cast_config_input.target_dtype,
                is_amax_initialized,
                reduce_amax=True,
            )
            input_fp8 = hp_tensor_to_float8_delayed(
                input,
                self.fp8_scale_input,
                self.config.cast_config_input.target_dtype,
                self.fp8_amax_input,
                linear_mm_config=self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )
        elif self.scaling_type_input is ScalingType.DYNAMIC:
            input_fp8 = hp_tensor_to_float8_dynamic(
                input,
                self.config.cast_config_input.target_dtype,
                self.linear_mm_config,
                gemm_input_role=GemmInputRole.INPUT,
            )
        else:
            assert self.scaling_type_input is ScalingType.STATIC
            input_fp8 = hp_tensor_to_float8_static(
                input,
                self.fp8_static_scale_input,
                self.config.cast_config_input.target_dtype,
                self.linear_mm_config,
            )

        return input_fp8

    def get_weight_scale(self, weight: torch.Tensor) -> Optional[torch.Tensor]:
        if tensor_already_casted_to_fp8(weight):
            return None
        if self.scaling_type_weight is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            _maybe_initialize_amaxes_scales_for_float8_cast(
                weight,
                self.fp8_amax_weight,
                self.fp8_amax_history_weight,
                self.fp8_scale_weight,
                scale_fn_name,
                self.config.cast_config_weight.target_dtype,
                self.is_amax_initialized,
                reduce_amax=True,
            )
            self.fp8_amax_weight.fill_(tensor_to_amax(weight))
            return self.fp8_scale_weight
        elif self.scaling_type_weight is ScalingType.DYNAMIC:
            return tensor_to_scale(weight, self.config.cast_config_weight.target_dtype)
        else:
            assert self.scaling_type_weight is ScalingType.STATIC
            return self.fp8_static_scale_weight

    def cast_output_to_float8_in_bw(self, output: torch.Tensor) -> torch.Tensor:
        if self.scaling_type_grad_output is ScalingType.DELAYED:
            scale_fn_name = self.config.delayed_scaling_config.scale_fn_name
            output = NoopFwToFloat8BwDelayed.apply(
                output,
                self.fp8_amax_grad_output,
                self.fp8_amax_history_grad_output,
                self.fp8_scale_grad_output,
                scale_fn_name,
                self.is_amax_initialized,
                self.linear_mm_config,
                self.config.cast_config_grad_output.target_dtype,
            )
        elif self.scaling_type_grad_output is ScalingType.DYNAMIC:
            output = NoopFwToFloat8BwDynamic.apply(
                output,
                self.linear_mm_config,
                self.config.cast_config_grad_output.target_dtype,
            )
        else:
            assert self.scaling_type_grad_output is ScalingType.STATIC
            output = NoopFwToFloat8BwStatic.apply(
                output,
                self.fp8_static_scale_grad_output,
                self.linear_mm_config,
                self.config.cast_config_grad_output.target_dtype,
            )
        return output

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.has_any_delayed_scaling:
            self.float8_pre_forward(input)
        output = super().forward(input)
        if self.has_any_delayed_scaling:
            self.float8_post_forward()
        return output

    def float8_pre_forward(self, input):
        # TODO(future PR): deprecate these functions and the corresponding
        # config setting
        if not self.enable_pre_and_post_forward:
            return

    def float8_post_forward(self):
        # TODO(future PR): deprecate these functions and the corresponding
        # config setting
        if not self.enable_pre_and_post_forward:
            return

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
        # need to create buffers again when moving from meta device to
        # real device
        new_mod.create_buffers()

        # If FSDP float8 all-gather is on, wrap the weight in a float8-aware
        # tensor subclass. This must happen last because:
        # 1. weight needs to be on the correct device to create the buffers
        # 2. buffers need to be already created for the delayed scaling version
        #    of the weight wrapper to be initialized
        if config.enable_fsdp_float8_all_gather:
            if config.cast_config_weight.scaling_type is ScalingType.DYNAMIC:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDynamicFloat8CastTensor(
                        new_mod.weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                    )
                )
            elif config.cast_config_weight.scaling_type is ScalingType.DELAYED:
                new_mod.weight = torch.nn.Parameter(
                    WeightWithDelayedFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_amax_weight,
                        new_mod.fp8_amax_history_weight,
                        new_mod.fp8_scale_weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                        new_mod.is_amax_initialized,
                    )
                )
            else:
                assert config.cast_config_weight.scaling_type is ScalingType.STATIC
                new_mod.weight = torch.nn.Parameter(
                    WeightWithStaticFloat8CastTensor(
                        new_mod.weight,
                        new_mod.fp8_static_scale_weight,
                        new_mod.linear_mm_config,
                        new_mod.config.cast_config_weight.target_dtype,
                    )
                )

        return new_mod

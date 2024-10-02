# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
A simple module swap UX for a float8 version of `torch.nn.Linear`.
"""

import logging

from typing import Optional

import torch

from torchao.float8.config import Float8CommLinearConfig, ScalingType
from torchao.float8.float8_tensor import Float8Tensor, LinearMMConfig

from torchao.float8.float8_comm_utils import (
    Float8CommWeightWithDelayedCastTensor,
    Float8CommWeightWithDynamicCastTensor,
    Float8CommWeightWithStaticCastTensor,
)

logger = logging.getLogger(__name__)


class Float8CommLinear(torch.nn.Linear):
    """
    Note: this is **not** a public API and is only intended to be used
    inside of this repository. Please file an issue if you would benefit
    from this being a public API.

    A wrapper around a `torch.nn.Linear` module which does fp8 compute, and tracks
    scales in way friendly to delayed scaling.
    """

    def __init__(self, *args, **kwargs):
        """
        Additional arguments on top of `torch.nn.Linear`'s arguments:
        * `config`: Float8CommLinearConfig
        """

        # Amax scales should always be kept as float32.
        self.always_float32_buffers = set()
        config = kwargs.pop("config")
        super().__init__(*args, **kwargs)

        # Defines the scaling behavior of weight
        self.scaling_type_weight = config.cast_config_weight.scaling_type
        # Convenience flag to skip code related to delayed scaling
        self.has_any_delayed_scaling = self.scaling_type_weight is ScalingType.DELAYED

        self.config = config

        self.create_buffers()

        # Note: is_amax_initialized is not a buffer to avoid data dependent
        # control flow visible to dynamo
        # TODO(future PR): add serialization for this flag
        self.is_amax_initialized = not self.config.enable_amax_init

        # Syncing of amaxes and scales happens outside of this function. This
        # flag is here to enforce that the user does not forget to do this.
        self.amax_and_scale_synced = not self.config.enable_amax_init

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
        # TODO(future PR): dtype values below don't have the other float8
        # flavors, fix it
        default_weight = torch.finfo(torch.float8_e4m3fn).max

        # Note: for now, create all the buffers if any are needed, to postpone
        # the work to make the scale and amax syncing and history calculation
        # handle a heterogeneous setup. We can do that work later if benchmarks
        # show it is worth doing.
        if self.has_any_delayed_scaling:
            self.register_always_float32_buffer(
                "fp8_amax_weight", torch.tensor([default_weight], device=device)
            )
            self.register_always_float32_buffer(
                "fp8_amax_history_weight", torch.zeros(history_len, device=device)
            )
            self.register_always_float32_buffer(
                "fp8_scale_weight", torch.tensor([1.0], device=device)
            )

        if self.config.cast_config_weight.static_scale is not None:
            self.register_always_float32_buffer(
                "fp8_static_scale_weight", 
                self.config.cast_config_weight.static_scale.to(device),
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

    def float8_pre_forward(self):
        if not self.enable_pre_and_post_forward:
            return
        if (
            self.is_amax_initialized
            and (not self.amax_and_scale_synced)
            and torch.is_grad_enabled()
        ):
            raise AssertionError(
                "amaxes and scales not synced, please call `sync_float8_amax_and_scale_history` before forward"
            )

    def float8_post_forward(self):
        if not self.enable_pre_and_post_forward:
            return
        # Ensure that calling forward again will fail until the user syncs
        # amaxes and scales
        self.is_amax_initialized = True
        self.amax_and_scale_synced = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.has_any_delayed_scaling:
            self.float8_pre_forward()

        # print("weight", self, self.weight)
        # output = super().forward(input)

        # assert isinstance(self.weight, Float8Tensor)
        # orig_weight = self.weight.to_original_precision()
        
        orig_weight = self.weight

        output = torch.matmul(input, orig_weight.t())
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)

        if self.has_any_delayed_scaling:
            self.float8_post_forward()
        return output

    def scaling_repr(self):
        # add scaling settings without using too many characters
        # example: "i:del,w:del,go:dyn"
        return f"w:{self.scaling_type_weight.short_str()}"

    def extra_repr(self):
        s = f'{super().extra_repr()}, scaling="{self.scaling_repr()}"'
        return s

    @classmethod
    def from_float(
        cls,
        mod,
        config: Optional[Float8CommLinearConfig] = None,
    ):
        """
        Create an nn.Linear with fp8 compute from a regular nn.Linear

        Args:
            mod (torch.nn.Linear): nn.Linear to convert
            config (Optional[Float8LinearConfig]): configuration for conversion to float8
        """
        if config is None:
            config = Float8CommLinearConfig()
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

        if config.cast_config_weight.scaling_type is ScalingType.DYNAMIC:
            new_mod.weight = torch.nn.Parameter(
                Float8CommWeightWithDynamicCastTensor(
                    new_mod.weight,
                )
            )
        elif config.cast_config_weight.scaling_type is ScalingType.DELAYED:
            new_mod.weight = torch.nn.Parameter(
                Float8CommWeightWithDelayedCastTensor(
                    new_mod.weight,
                    new_mod.fp8_amax_weight,
                    new_mod.fp8_amax_history_weight,
                    new_mod.fp8_scale_weight,
                    new_mod.is_amax_initialized,
                )
            )
        else:
            assert config.cast_config_weight.scaling_type is ScalingType.STATIC
            new_mod.weight = torch.nn.Parameter(
                Float8CommWeightWithStaticCastTensor(
                    new_mod.weight,
                    new_mod.fp8_static_scale_weight,
                )
            )

        return new_mod

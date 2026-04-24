# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
NVFP4 training configuration and linear module.

Provides NVFP4TrainingConfig for use with quantize_() and an
NVFP4Linear module that performs NVFP4 quantized GEMMs
in both forward and backward passes.

Usage:
    from torchao.prototype.mx_formats.nvfp4_training import NVFP4TrainingConfig
    from torchao.quantization import quantize_

    quantize_(model, NVFP4TrainingConfig())
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.mx_formats.nvfp4_linear import nvfp4_linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler


@dataclass
class NVFP4TrainingConfig(AOBaseConfig):
    """Configuration for NVFP4 quantized training.

    When passed to quantize_(), replaces nn.Linear modules with
    NVFP4Linear, which quantizes all three GEMMs (forward
    and backward) to NVFP4.

    Args:
        kernel_preference: Backend for quantization kernels.
            TRITON: Pure-Triton RHT + stochastic rounding path.
            Default: TRITON.
        process_group: Optional ProcessGroup for tensor-parallel TP.
            When set with kernel_preference=TRITON, forward dispatches to
            the selected NVFP4 tensor-parallel path.
        world_size: TP world size.  Inferred from process_group if None.
    """

    kernel_preference: KernelPreference = KernelPreference.TRITON
    process_group: Optional[object] = field(default=None, compare=False)
    world_size: Optional[int] = None


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 quantized forward and backward GEMMs.

    Drop-in replacement for nn.Linear that quantizes activations, weights,
    and gradients to NVFP4 for all three training GEMMs.

    When process_group is set and kernel_preference==TRITON the forward uses
    the tensor-parallel protocol selected by NVFP4ColwiseParallel or
    NVFP4RowwiseParallel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        kernel_preference: KernelPreference = KernelPreference.TRITON,
        process_group=None,
        world_size: Optional[int] = None,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.kernel_preference = kernel_preference
        self.process_group = process_group
        self.world_size = world_size
        self.tensor_parallel_style = "colwise"
        self._sr_seed: Optional[torch.Tensor] = None

    def _ensure_sr_seed(self, device: torch.device | str) -> torch.Tensor:
        if self._sr_seed is None:
            self._sr_seed = torch.randint(
                -(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device
            )
        return self._sr_seed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.process_group is not None
            and self.kernel_preference == KernelPreference.TRITON
        ):
            import torch.distributed as dist
            from torch.distributed.tensor import DTensor

            from torchao.prototype.mx_formats.nvfp4_tensor_parallel import (
                nvfp4_col_parallel_linear,
                nvfp4_row_parallel_linear,
            )

            ws = self.world_size
            if ws is None:
                ws = dist.get_world_size(self.process_group)
            sr_seed = self._ensure_sr_seed(x.device)
            w = self.weight
            if isinstance(w, DTensor):
                w = w.to_local()
            bias = self.bias
            if isinstance(bias, DTensor):
                bias = bias.to_local()
            tp_linear = (
                nvfp4_row_parallel_linear
                if self.tensor_parallel_style == "rowwise"
                else nvfp4_col_parallel_linear
            )
            return tp_linear(
                x,
                w,
                bias,
                sr_seed=sr_seed,
                tp_group=self.process_group,
                world_size=ws,
            )
        return nvfp4_linear(
            x, self.weight, self.bias, kernel_preference=self.kernel_preference
        )

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        kernel_preference: KernelPreference = KernelPreference.TRITON,
        process_group=None,
        world_size: Optional[int] = None,
    ) -> "NVFP4Linear":
        new = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            kernel_preference=kernel_preference,
            process_group=process_group,
            world_size=world_size,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
        )
        # Copy weights (don't re-init)
        if mod.weight.device != torch.device("meta"):
            new.weight = mod.weight
            if mod.bias is not None:
                new.bias = mod.bias
        return new


@register_quantize_module_handler(NVFP4TrainingConfig)
def _nvfp4_training_transform(
    module: nn.Module,
    config: NVFP4TrainingConfig,
    parameter_name: Optional[str] = None,
) -> nn.Module:
    """Handler for quantize_(): replaces nn.Linear with NVFP4Linear."""
    if isinstance(module, nn.Linear):
        return NVFP4Linear.from_linear(
            module,
            kernel_preference=config.kernel_preference,
            process_group=config.process_group,
            world_size=config.world_size,
        )
    return module

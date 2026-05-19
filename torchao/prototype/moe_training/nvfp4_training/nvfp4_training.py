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
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_training import NVFP4TrainingConfig
    from torchao.quantization import quantize_

    quantize_(model, NVFP4TrainingConfig())
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.moe_training.nvfp4_training.hadamard_utils import (
    get_wgrad_sign_vector,
)
from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import nvfp4_linear
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.quantization.transform_module import register_quantize_module_handler


def _rht_sign_vector_to_tuple(sign_vector: torch.Tensor) -> tuple[int, ...] | None:
    if hasattr(sign_vector, "to_local"):
        sign_vector = sign_vector.to_local()
    if sign_vector.device.type == "meta":
        return None
    return tuple(int(v) for v in sign_vector.detach().cpu().tolist())


def _make_rht_sign_vector(
    sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None,
    device,
) -> torch.Tensor:
    if sign_vector is None:
        if device is not None and torch.device(device).type == "meta":
            return torch.empty(16, dtype=torch.int8, device=device)
        return get_wgrad_sign_vector(16, device=device, dtype=torch.int8)

    if isinstance(sign_vector, torch.Tensor):
        if sign_vector.numel() != 16:
            raise ValueError(
                f"Expected rht_sign_vector length 16, got {sign_vector.numel()}"
            )
        kwargs = {"dtype": torch.int8}
        if device is not None:
            kwargs["device"] = device
        return sign_vector.detach().to(**kwargs).clone()

    if len(sign_vector) != 16:
        raise ValueError(f"Expected rht_sign_vector length 16, got {len(sign_vector)}")
    return torch.tensor(sign_vector, dtype=torch.int8, device=device)


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
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.kernel_preference = kernel_preference
        self.process_group = process_group
        self.world_size = world_size
        self.tensor_parallel_style = "colwise"
        self.register_buffer(
            "_sr_seed",
            torch.randint(-(2**63), 2**63 - 1, (1,), dtype=torch.int64, device=device),
        )
        self.register_buffer(
            "_rht_sign_vector",
            _make_rht_sign_vector(rht_sign_vector, device=device),
            persistent=True,
        )
        self._refresh_rht_sign_vector_tuple()

    def _refresh_rht_sign_vector_tuple(self) -> None:
        self._rht_sign_vector_tuple = _rht_sign_vector_to_tuple(self._rht_sign_vector)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_rht_sign_vector_tuple()

    @property
    def rht_sign_vector(self) -> tuple[int, ...]:
        if self._rht_sign_vector_tuple is None:
            self._refresh_rht_sign_vector_tuple()
        if self._rht_sign_vector_tuple is None:
            raise RuntimeError("rht_sign_vector is not materialized")
        return self._rht_sign_vector_tuple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.process_group is not None
            and self.kernel_preference == KernelPreference.TRITON
        ):
            import torch.distributed as dist
            from torch.distributed.tensor import DTensor

            from torchao.prototype.moe_training.nvfp4_training.nvfp4_tensor_parallel import (
                nvfp4_col_parallel_linear,
                nvfp4_row_parallel_linear,
            )

            ws = self.world_size
            if ws is None:
                ws = dist.get_world_size(self.process_group)
            sr_seed = self._sr_seed
            if isinstance(sr_seed, DTensor):
                sr_seed = sr_seed.to_local()
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
            x,
            self.weight,
            self.bias,
            kernel_preference=self.kernel_preference,
            sr_seed=self._sr_seed,
            sign_vector=self.rht_sign_vector,
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

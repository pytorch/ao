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
                f"rht_sign_vector must have 16 elements, got {sign_vector.numel()}"
            )
        kwargs = {"dtype": torch.int8}
        if device is not None:
            kwargs["device"] = device
        return sign_vector.detach().to(**kwargs).clone()

    if len(sign_vector) != 16:
        raise ValueError(
            f"rht_sign_vector must have 16 elements, got {len(sign_vector)}"
        )
    return torch.tensor(sign_vector, dtype=torch.int8, device=device)


@dataclass
class NVFP4TrainingConfig(AOBaseConfig):
    """Configuration for NVFP4 quantized training.

    When passed to quantize_(), replaces nn.Linear modules with the module
    implementing the selected recipe, which quantizes all three GEMMs
    (forward and backward) to NVFP4.

    Args:
        kernel_preference: ("default" recipe) Backend for quantization kernels.
            TRITON: Pure-Triton RHT + stochastic rounding path.
            CUTEDSL: CuteDSL kernels for the full quantize path (amax, forward
                RTNE quantize, SR backward quantize, and 2D weight quantize).
                Requires SM100; in_features divisible by 128 and out_features
                by 256. Under tensor parallel the same constraints apply to each
                per-rank shard, and the per-rank M shard must be divisible by 256.
            Default: TRITON.
        process_group: ("default" recipe) Optional ProcessGroup for
            tensor-parallel TP.
            When set, forward dispatches to the selected NVFP4 tensor-parallel
            path (TRITON or CUTEDSL).
        world_size: ("default" recipe) TP world size.  Inferred from
            process_group if None.
        rht_sign_vector: ("default" recipe) Optional {-1, 1} sign vector of length 16 for the
            randomized Hadamard transform.  When None, each NVFP4Linear draws
            its own random vector.  In multi-rank settings (FSDP) replicas will
            therefore have different bases — harmless for convergence but
            inconsistent across checkpoints.  Callers that require replica
            consistency should broadcast a single vector before calling
            quantize_() and pass it here.  The TP path always enforces
            consistency via _replicate_rht_sign_vector regardless of this field.
        recipe: Which NVFP4 training recipe to install.
            "default": NVFP4Linear — randomized Hadamard transform + stochastic
                rounding, configured by the fields above.
            "four_over_six": NVFP4FourOverSixLinear — adaptive per-block
                candidate selection between the standard map-to-6 encoding and
                a 1.5x-scale map-to-4 encoding, configured by the fields below.
                Stateless (no RHT/SR buffers, no TP support).
            Each recipe's fields must stay at their defaults under the other
            recipe.
        err_mode: ("four_over_six" recipe) Candidate-selection error metric,
            "mae" or "mse".
        e4m3_scale_bound: ("four_over_six" recipe) Global E4M3 scale bound;
            256 leaves map-to-4 headroom, 448 uses the full range.
        row_scaled_activation: ("four_over_six" recipe) Derive one FP32 global
            scale per activation row instead of per tensor.
        backward_override: ("four_over_six" recipe) Backward mode — None
            (recipe default), "quantized", "high_precision", or "dequantized".
        weight_block: ("four_over_six" recipe) Weight block granularity,
            "16x16" or "1x16".
    """

    kernel_preference: KernelPreference = KernelPreference.TRITON
    process_group: Optional[object] = field(default=None, compare=False)
    world_size: Optional[int] = None
    rht_sign_vector: Optional[object] = field(default=None, compare=False)
    recipe: str = "default"
    err_mode: str = "mae"
    e4m3_scale_bound: int = 256
    row_scaled_activation: bool = False
    backward_override: Optional[str] = None
    weight_block: str = "16x16"

    def __post_init__(self):
        if self.recipe not in ("default", "four_over_six"):
            raise ValueError(
                f"recipe must be 'default' or 'four_over_six', got {self.recipe!r}"
            )
        if self.recipe == "default":
            four_over_six_defaults = (
                ("err_mode", self.err_mode, "mae"),
                ("e4m3_scale_bound", self.e4m3_scale_bound, 256),
                ("row_scaled_activation", self.row_scaled_activation, False),
                ("backward_override", self.backward_override, None),
                ("weight_block", self.weight_block, "16x16"),
            )
            for name, value, default in four_over_six_defaults:
                if value != default:
                    raise ValueError(
                        f"{name} configures the 'four_over_six' recipe and must "
                        f"stay at its default under recipe='default', got {value!r}"
                    )
            return
        if self.err_mode not in ("mae", "mse"):
            raise ValueError(f"err_mode must be 'mae' or 'mse', got {self.err_mode!r}")
        if self.e4m3_scale_bound not in (256, 448):
            raise ValueError(
                f"e4m3_scale_bound must be 256 or 448, got {self.e4m3_scale_bound}"
            )
        if self.backward_override not in (
            None,
            "quantized",
            "high_precision",
            "dequantized",
        ):
            raise ValueError(
                f"backward_override must be None, 'quantized', 'high_precision', "
                f"or 'dequantized', got {self.backward_override!r}"
            )
        if self.weight_block not in ("1x16", "16x16"):
            raise ValueError(
                f"weight_block must be '1x16' or '16x16', got {self.weight_block!r}"
            )
        default_recipe_defaults = (
            ("kernel_preference", self.kernel_preference, KernelPreference.TRITON),
            ("process_group", self.process_group, None),
            ("world_size", self.world_size, None),
            ("rht_sign_vector", self.rht_sign_vector, None),
        )
        for name, value, default in default_recipe_defaults:
            if value is not default and value != default:
                raise ValueError(
                    f"{name} configures the 'default' recipe and must stay at "
                    f"its default under recipe='four_over_six', got {value!r}"
                )


class NVFP4Linear(nn.Linear):
    """Linear layer with NVFP4 quantized forward and backward GEMMs.

    Drop-in replacement for nn.Linear that quantizes activations, weights,
    and gradients to NVFP4 for all three training GEMMs.

    When process_group is set the forward uses the tensor-parallel protocol
    selected by NVFP4ColwiseParallel or NVFP4RowwiseParallel.
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
        if self.process_group is not None and self.kernel_preference in (
            KernelPreference.TRITON,
            KernelPreference.CUTEDSL,
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
                sign_vector=self.rht_sign_vector,
                use_cutedsl=self.kernel_preference == KernelPreference.CUTEDSL,
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
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
    ) -> "NVFP4Linear":
        if rht_sign_vector is None:
            rht_sign_vector = getattr(mod, "_rht_sign_vector", None)
        new = cls(
            mod.in_features,
            mod.out_features,
            mod.bias is not None,
            kernel_preference=kernel_preference,
            process_group=process_group,
            world_size=world_size,
            device=mod.weight.device,
            dtype=mod.weight.dtype,
            rht_sign_vector=rht_sign_vector,
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
    """Handler for quantize_(): replaces nn.Linear with config.recipe's module.

    Modules already converted to either recipe are left alone.
    """
    from torchao.prototype.moe_training.nvfp4_training.four_over_six import (
        NVFP4FourOverSixLinear,
    )

    if isinstance(module, (NVFP4Linear, NVFP4FourOverSixLinear)):
        return module
    if not isinstance(module, nn.Linear):
        return module
    if config.recipe == "four_over_six":
        return NVFP4FourOverSixLinear.from_linear(
            module,
            err_mode=config.err_mode,
            e4m3_scale_bound=config.e4m3_scale_bound,
            row_scaled_activation=config.row_scaled_activation,
            backward_override=config.backward_override,
            weight_block=config.weight_block,
        )
    return NVFP4Linear.from_linear(
        module,
        kernel_preference=config.kernel_preference,
        process_group=config.process_group,
        world_size=config.world_size,
        rht_sign_vector=config.rht_sign_vector,
    )

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

    When passed to quantize_(), replaces nn.Linear modules with
    NVFP4Linear, which quantizes all three GEMMs (forward
    and backward) to NVFP4.

    Args:
        kernel_preference: Backend for quantization kernels.
            AUTO: CuteDSL where its runtime and the shapes allow, Triton otherwise.
                The tensor-parallel path is the exception and stays on Triton, since
                its shard constraints are checked by raising rather than by a
                predicate AUTO could fall back on.
            TRITON: Pure-Triton RHT + stochastic rounding path.
            CUTEDSL: CuteDSL kernels for the full quantize path (amax, forward
                RTNE quantize, SR backward quantize, and 2D weight quantize).
                Requires SM100; in_features divisible by 128 and out_features
                by 128. Under tensor parallel the same constraints apply to each
                per-rank shard, and the per-rank M shard must be divisible by 128.
                Unlike AUTO, an unmet requirement raises instead of falling back.
            Default: AUTO.

            Reproducibility note. AUTO resolves per call site from what the runtime
            offers, so the backend can differ between two nodes running the same code.
            Under stochastic rounding that changes results: the CuteDSL and Triton
            kernels are byte-identical under RTNE but draw *different*
            stochastic-rounding streams (CuteDSL takes one Philox counter per
            16-element block and consumes all four words, rather than reproducing
            Triton's per-packed-byte counter stride). This holds on the **linear and
            grouped paths alike** -- both draw through ``philox4_all``. SR runs in the
            backward pass, so the same seed on a node without the CuteDSL runtime
            yields different gradients -- statistically equivalent, not bitwise equal.
            Pin kernel_preference explicitly for runs that must reproduce bitwise
            across machines.
        process_group: Optional ProcessGroup for tensor-parallel TP.
            When set, forward dispatches to the selected NVFP4 tensor-parallel
            path (Triton unless CUTEDSL is requested explicitly).
        world_size: TP world size.  Inferred from process_group if None.
        rht_sign_vector: Optional {-1, 1} sign vector of length 16 for the
            randomized Hadamard transform.  When None, each NVFP4Linear draws
            its own random vector.  In multi-rank settings (FSDP) replicas will
            therefore have different bases — harmless for convergence but
            inconsistent across checkpoints.  Callers that require replica
            consistency should broadcast a single vector before calling
            quantize_() and pass it here.  The TP path always enforces
            consistency via _replicate_rht_sign_vector regardless of this field.
        use_fast_math: Match TransformerEngine under ``NVTE_USE_FAST_MATH=1``: the RHT
            quantize consumes the FP32 accumulator directly and takes an approximate
            reciprocal. On by default; both backends implement it and remain bitwise
            identical to TE and to each other. Set False to recover the exact-math
            arithmetic.  Default: True.

    Both defaults moved together, and both change what this config computes:
    ``kernel_preference`` was TRITON and is now AUTO, and ``use_fast_math`` is new and
    defaults on (fast-vs-exact measures 30-32 dB SQNR columnwise -- above NVFP4's own
    ~20 dB quantization noise, but not identical). A run that must reproduce earlier
    numerics needs both pinned::

        NVFP4TrainingConfig(
            kernel_preference=KernelPreference.TRITON, use_fast_math=False
        )
    """

    kernel_preference: KernelPreference = KernelPreference.AUTO
    process_group: Optional[object] = field(default=None, compare=False)
    world_size: Optional[int] = None
    rht_sign_vector: Optional[object] = field(default=None, compare=False)
    use_fast_math: bool = True


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
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        process_group=None,
        world_size: Optional[int] = None,
        device=None,
        dtype=None,
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
        use_fast_math: bool = True,
    ):
        super().__init__(in_features, out_features, bias, device=device, dtype=dtype)
        self.kernel_preference = kernel_preference
        self.use_fast_math = use_fast_math
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
            KernelPreference.AUTO,
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
                # CUTEDSL only, not AUTO: the TP path states its shard requirements by
                # raising (_check_cutedsl_shard), so there is no predicate for AUTO to
                # fall back on and a misaligned shard would turn a working default into
                # an error. TP stays on Triton until that check is available as a bool.
                use_cutedsl=self.kernel_preference == KernelPreference.CUTEDSL,
                use_fast_math=self.use_fast_math,
            )
        return nvfp4_linear(
            x,
            self.weight,
            self.bias,
            kernel_preference=self.kernel_preference,
            sr_seed=self._sr_seed,
            sign_vector=self.rht_sign_vector,
            use_fast_math=self.use_fast_math,
        )

    @classmethod
    def from_linear(
        cls,
        mod: nn.Linear,
        kernel_preference: KernelPreference = KernelPreference.AUTO,
        process_group=None,
        world_size: Optional[int] = None,
        rht_sign_vector: torch.Tensor | tuple[int, ...] | list[int] | None = None,
        use_fast_math: bool = True,
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
            use_fast_math=use_fast_math,
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
    if isinstance(module, NVFP4Linear):
        return module
    if isinstance(module, nn.Linear):
        return NVFP4Linear.from_linear(
            module,
            kernel_preference=config.kernel_preference,
            process_group=config.process_group,
            world_size=config.world_size,
            rht_sign_vector=config.rht_sign_vector,
            use_fast_math=config.use_fast_math,
        )
    return module

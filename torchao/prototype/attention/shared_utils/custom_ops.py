# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared custom op registration and compile helpers for FP8 attention backends.

Custom ops are needed because our FP8 SDPA functions call Triton kernels
that are not traceable by torch.compile. Registering them as custom_ops
tells the compiler to treat them as opaque nodes with known shapes/dtypes.
"""

from functools import partial
from typing import Callable, NamedTuple

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.fusion_utils import (
    detect_causal_mask,
)
from torchao.prototype.attention.shared_utils.fusion_utils import (
    rope_sdpa_fusion_pass as _shared_fusion_pass,
)


class RegisteredOps(NamedTuple):
    rope_sdpa_op: object
    fp8_sdpa_op: object


def register_fp8_attention_ops(
    backend_name: str,
    rope_sdpa_fn: Callable,
    sdpa_fn: Callable,
) -> RegisteredOps:
    """Register the RoPE+SDPA and plain SDPA custom ops for a backend."""
    backend = backend_name.lower()

    rope_op_name = f"torchao::fp8_{backend}_rope_sdpa"

    @torch.library.custom_op(rope_op_name, mutates_args=())
    def _rope_sdpa_custom_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = False,
        scale: float = 0.0,
        enable_gqa: bool = False,
        rope_interleaved: bool = False,
    ) -> torch.Tensor:
        actual_scale = scale if scale != 0.0 else None
        return rope_sdpa_fn(
            q,
            k,
            v,
            cos,
            sin,
            is_causal=is_causal,
            scale=actual_scale,
            enable_gqa=enable_gqa,
            rope_interleaved=rope_interleaved,
        )

    @_rope_sdpa_custom_op.register_fake
    def _rope_sdpa_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = False,
        scale: float = 0.0,
        enable_gqa: bool = False,
        rope_interleaved: bool = False,
    ) -> torch.Tensor:
        B, S, H, D = q.shape
        return torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)

    sdpa_op_name = f"torchao::fp8_{backend}_sdpa"

    @torch.library.custom_op(sdpa_op_name, mutates_args=())
    def _sdpa_custom_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        scale: float = 0.0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        actual_scale = scale if scale != 0.0 else None
        return sdpa_fn(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=actual_scale,
            enable_gqa=enable_gqa,
        )

    @_sdpa_custom_op.register_fake
    def _sdpa_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        scale: float = 0.0,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    rope_sdpa_op = getattr(
        getattr(torch.ops, "torchao"), f"fp8_{backend}_rope_sdpa"
    ).default
    fp8_sdpa_op = getattr(getattr(torch.ops, "torchao"), f"fp8_{backend}_sdpa").default

    return RegisteredOps(rope_sdpa_op=rope_sdpa_op, fp8_sdpa_op=fp8_sdpa_op)


def make_backend_fn(
    ops: RegisteredOps,
    backend_name: str,
    flash_impl_name: str,
    max_head_dim: int = 256,
) -> Callable:
    """Return a ``make_fp8_backend(model, config)`` function for a backend."""

    def make_fp8_backend(
        model: nn.Module,
        config: LowPrecisionAttentionConfig,
    ) -> Callable:
        from torch._inductor.compile_fx import compile_fx

        strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)

        pass_fn = partial(
            _shared_fusion_pass,
            rope_sdpa_op=ops.rope_sdpa_op,
            fp8_sdpa_op=ops.fp8_sdpa_op,
            max_head_dim=max_head_dim,
            backend_name=backend_name,
            fuse_rope=config.fuse_rope_using_torch_compile,
            strip_causal_mask=strip_causal_mask,
        )

        def fp8_attention_backend(gm, example_inputs):
            old_pass = inductor_config.pre_grad_custom_pass
            inductor_config.pre_grad_custom_pass = pass_fn
            try:
                return compile_fx(gm, example_inputs)
            finally:
                inductor_config.pre_grad_custom_pass = old_pass

        return fp8_attention_backend

    return make_fp8_backend

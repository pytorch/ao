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

from typing import Callable, NamedTuple

import torch


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
        hadamard: str = "NONE",
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
            hadamard=hadamard,
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
        hadamard: str = "NONE",
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
        hadamard: str = "NONE",
    ) -> torch.Tensor:
        actual_scale = scale if scale != 0.0 else None
        return sdpa_fn(
            q,
            k,
            v,
            is_causal=is_causal,
            scale=actual_scale,
            enable_gqa=enable_gqa,
            hadamard=hadamard,
        )

    @_sdpa_custom_op.register_fake
    def _sdpa_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        scale: float = 0.0,
        enable_gqa: bool = False,
        hadamard: str = "NONE",
    ) -> torch.Tensor:
        return torch.empty(q.shape, dtype=q.dtype, device=q.device)

    rope_sdpa_op = getattr(
        getattr(torch.ops, "torchao"), f"fp8_{backend}_rope_sdpa"
    ).default
    fp8_sdpa_op = getattr(getattr(torch.ops, "torchao"), f"fp8_{backend}_sdpa").default

    return RegisteredOps(rope_sdpa_op=rope_sdpa_op, fp8_sdpa_op=fp8_sdpa_op)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared custom op registration and compile helpers for FP8 attention backends.

Provides ``register_fp8_attention_ops`` which registers the two custom ops
needed by each backend (rope+SDPA and plain SDPA), and
``make_compile_fn`` which builds the ``compile_with_fp8_fusion`` callable.

Backend-specific modules (e.g., ``fp8_fa3/fusion_pass.py``) call these
once at import time and then expose thin entry points.

The custom ops are necessary because the FP8 SDPA functions call Triton
kernels for the fused RoPE + FP8 quantization step. These Triton kernels
are not traceable by torch.compile — they aren't built from standard
PyTorch ops that the compiler can decompose. Registering them as
``torch.library.custom_op`` tells the compiler to treat them as opaque
nodes with known output shapes/dtypes (via ``register_fake``).
"""

from functools import partial
from typing import Callable, NamedTuple

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn
from torch.fx import Graph

from torchao.prototype.attention.config import LowPrecisionAttentionConfig
from torchao.prototype.attention.shared_utils.fusion_utils import (
    detect_causal_mask,
)
from torchao.prototype.attention.shared_utils.fusion_utils import (
    rope_sdpa_fusion_pass as _shared_fusion_pass,
)

_CUSTOM_OP_LIB = "torchao"


class RegisteredOps(NamedTuple):
    """The two custom op references returned by ``register_fp8_attention_ops``."""

    rope_sdpa_op: object  # e.g. torch.ops.torchao.fp8_fa3_rope_sdpa.default
    fp8_sdpa_op: object  # e.g. torch.ops.torchao.fp8_fa3_sdpa.default


def register_fp8_attention_ops(
    backend_name: str,
    rope_sdpa_fn: Callable,
    sdpa_fn: Callable,
) -> RegisteredOps:
    """Register the RoPE+SDPA and plain SDPA custom ops for a backend.

    This creates two ``torch.library.custom_op`` entries in the
    ``torchao`` namespace:

    - ``torchao::fp8_{backend}_rope_sdpa`` — fused RoPE + FP8 SDPA
    - ``torchao::fp8_{backend}_sdpa`` — plain FP8 SDPA (no RoPE)

    Each op has a ``register_fake`` implementation so that ``torch.compile``
    can infer output shapes/dtypes without running the real kernels.

    Args:
        backend_name: Lowercase backend identifier (e.g. ``"fa3"``, ``"fa4"``).
        rope_sdpa_fn: The backend-specific fused RoPE+SDPA function
            (e.g. ``fp8_fa3_rope_sdpa``).
        sdpa_fn: The backend-specific plain SDPA function
            (e.g. ``fp8_fa3_sdpa``).

    Returns:
        A ``RegisteredOps`` namedtuple with the two resolved op references
        (suitable for passing to the shared fusion pass).
    """
    backend = backend_name.lower()

    # ------------------------------------------------------------------
    # Rope + SDPA custom op
    # ------------------------------------------------------------------
    rope_op_name = f"{_CUSTOM_OP_LIB}::fp8_{backend}_rope_sdpa"

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

    # ------------------------------------------------------------------
    # Plain SDPA custom op
    # ------------------------------------------------------------------
    sdpa_op_name = f"{_CUSTOM_OP_LIB}::fp8_{backend}_sdpa"

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

    # ------------------------------------------------------------------
    # Resolve the registered ops
    # ------------------------------------------------------------------
    rope_sdpa_op = getattr(
        getattr(torch.ops, _CUSTOM_OP_LIB), f"fp8_{backend}_rope_sdpa"
    ).default
    fp8_sdpa_op = getattr(
        getattr(torch.ops, _CUSTOM_OP_LIB), f"fp8_{backend}_sdpa"
    ).default

    return RegisteredOps(rope_sdpa_op=rope_sdpa_op, fp8_sdpa_op=fp8_sdpa_op)


def make_fusion_pass(
    ops: RegisteredOps,
    backend_name: str,
    max_head_dim: int = 256,
) -> Callable:
    """Create a backend-specific fusion pass function.

    Returns a callable with signature
    ``(graph, *, fuse_rope=True, strip_causal_mask=False) -> None``
    that delegates to the shared ``rope_sdpa_fusion_pass`` with the
    backend's custom ops.

    Args:
        ops: The registered custom ops from ``register_fp8_attention_ops``.
        backend_name: Backend name for logging (e.g. ``"FA3"``).
        max_head_dim: Maximum supported head dimension.

    Returns:
        The fusion pass callable.
    """

    def _fusion_pass(
        graph: Graph, *, fuse_rope: bool = True, strip_causal_mask: bool = False
    ) -> None:
        _shared_fusion_pass(
            graph,
            rope_sdpa_op=ops.rope_sdpa_op,
            fp8_sdpa_op=ops.fp8_sdpa_op,
            max_head_dim=max_head_dim,
            backend_name=backend_name,
            fuse_rope=fuse_rope,
            strip_causal_mask=strip_causal_mask,
        )

    return _fusion_pass


def make_compile_fn(
    fusion_pass_fn: Callable,
    flash_impl_name: str,
) -> Callable:
    """Create a ``compile_with_fp8_fusion`` function for a backend.

    Args:
        fusion_pass_fn: The backend-specific fusion pass (from
            ``make_fusion_pass``).
        flash_impl_name: Flash implementation name (e.g. ``"FA3"``).

    Returns:
        A callable ``(model, config) -> compiled_module``.
    """

    def compile_with_fp8_fusion(
        model: nn.Module,
        config: LowPrecisionAttentionConfig,
    ) -> nn.Module:
        from torch._inductor.compile_fx import compile_fx

        strip_causal_mask = detect_causal_mask(model, flash_impl_name=flash_impl_name)

        pass_fn = partial(
            fusion_pass_fn,
            fuse_rope=config.fuse_rope,
            strip_causal_mask=strip_causal_mask,
        )

        def fp8_attention_backend(gm, example_inputs):
            old_pass = inductor_config.pre_grad_custom_pass
            inductor_config.pre_grad_custom_pass = pass_fn
            try:
                return compile_fx(gm, example_inputs)
            finally:
                inductor_config.pre_grad_custom_pass = old_pass

        torch._dynamo.reset()
        return torch.compile(model, backend=fp8_attention_backend)

    return compile_with_fp8_fusion

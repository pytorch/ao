# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FA4-specific FX graph fusion pass.

Registers FA4 custom ops (torchao::fp8_fa4_rope_sdpa, torchao::fp8_fa4_sdpa)
and provides a thin wrapper around the shared fusion pass that uses them.

Pattern detection, graph surgery, and the main fusion loop are in
torchao.prototype.attention.fusion_utils.
"""

import torch
from torch.fx import Graph

from torchao.prototype.attention.fusion_utils import (
    rope_sdpa_fusion_pass as _shared_fusion_pass,
)


# ============================================================================
# Custom Op Registration (FA4-specific)
# ============================================================================
#
# Same pattern as FA3: register custom ops so torch.compile can embed
# them as opaque nodes in the FX graph. The underlying implementations
# call the FA4 attention functions which activate the FA4 backend.

_CUSTOM_OP_LIB = "torchao"
_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa4_rope_sdpa"


@torch.library.custom_op(_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa4_rope_sdpa_custom_op(
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
    """Custom op wrapper around fp8_fa4_rope_sdpa.

    Args:
        q: Query tensor [B, S, H, D] in bf16/fp16.
        k: Key tensor [B, S, H, D] in bf16/fp16.
        v: Value tensor [B, S, H, D] in bf16/fp16.
        cos: RoPE cosine frequencies [S, D].
        sin: RoPE sine frequencies [S, D].
        is_causal: Whether to use causal masking.
        scale: Attention scale factor. 0.0 = use default (1/sqrt(D)).
        enable_gqa: Whether to enable grouped query attention.
        rope_interleaved: Whether to use interleaved (FLUX) or NeoX (LLaMA) RoPE.

    Returns:
        Attention output [B, H, S, D] in the input dtype.
    """
    from torchao.prototype.attention.fp8_fa4.attention import fp8_fa4_rope_sdpa

    actual_scale = scale if scale != 0.0 else None

    return fp8_fa4_rope_sdpa(
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


@_fp8_fa4_rope_sdpa_custom_op.register_fake
def _fp8_fa4_rope_sdpa_fake(
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
    """FakeTensor implementation: output shape [B, H, S, D] from input [B, S, H, D]."""
    B, S, H, D = q.shape
    return torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)


_NON_ROPE_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa4_sdpa"


@torch.library.custom_op(_NON_ROPE_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa4_sdpa_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Custom op wrapper around fp8_fa4_sdpa (non-rope version).

    Args:
        q: Query tensor [B, H, S, D] in bf16/fp16.
        k: Key tensor [B, H, S, D] in bf16/fp16.
        v: Value tensor [B, H, S, D] in bf16/fp16.
        is_causal: Whether to use causal masking.
        scale: Attention scale factor. 0.0 = use default (1/sqrt(D)).
        enable_gqa: Whether to enable grouped query attention.

    Returns:
        Attention output [B, H, S, D] in the input dtype.
    """
    from torchao.prototype.attention.fp8_fa4.attention import fp8_fa4_sdpa

    actual_scale = scale if scale != 0.0 else None

    return fp8_fa4_sdpa(
        q,
        k,
        v,
        is_causal=is_causal,
        scale=actual_scale,
        enable_gqa=enable_gqa,
    )


@_fp8_fa4_sdpa_custom_op.register_fake
def _fp8_fa4_sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """FakeTensor implementation for non-rope FP8 SDPA. Input/output: [B, H, S, D]."""
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ============================================================================
# FA4-specific Fusion Pass Entry Point
# ============================================================================


def rope_sdpa_fusion_pass(graph: Graph) -> None:
    """FA4-specific fusion pass: detects and replaces SDPA patterns with FA4 ops.

    This is the entry point registered as a pre-grad custom pass via
    torch._inductor.config.pre_grad_custom_pass. It delegates to the shared
    fusion pass with FA4-specific custom ops.

    Args:
        graph: The FX graph to transform (modified in-place).
    """
    _shared_fusion_pass(
        graph,
        rope_sdpa_op=torch.ops.torchao.fp8_fa4_rope_sdpa.default,
        fp8_sdpa_op=torch.ops.torchao.fp8_fa4_sdpa.default,
        max_head_dim=256,
        backend_name="FA4",
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FA3-specific FX graph fusion pass.

Registers FA3 custom ops (torchao::fp8_fa3_rope_sdpa, torchao::fp8_fa3_sdpa)
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
# Custom Op Registration (FA3-specific)
# ============================================================================
#
# The fp8_fa3_rope_sdpa function in attention.py calls Triton kernels for
# the fused RoPE + FP8 quantization step (_fp8_rope_sdpa_quantize). These
# Triton kernels are not traceable by torch.compile -- they aren't built
# from standard PyTorch ops that the compiler can decompose.
#
# To make fp8_fa3_rope_sdpa usable inside a compiled graph, we register it
# as a torch.library.custom_op. This tells the compiler:
#   1. "This is an opaque operation -- don't try to trace into it."
#   2. "Here's how to compute output shapes/dtypes" (via register_fake).
#   3. "Here's the real implementation to call at runtime."
#
# The custom op is what we insert into the FX graph during fusion.
# After registration, it's accessible as torch.ops.torchao.fp8_fa3_rope_sdpa.

_CUSTOM_OP_LIB = "torchao"
_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa3_rope_sdpa"


@torch.library.custom_op(_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa3_rope_sdpa_custom_op(
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
    """Custom op wrapper around fp8_fa3_rope_sdpa.

    Args:
        q: Query tensor [B, S, H, D] in bf16/fp16.
        k: Key tensor [B, S, H, D] in bf16/fp16.
        v: Value tensor [B, S, H, D] in bf16/fp16.
        cos: RoPE cosine frequencies [S, D].
        sin: RoPE sine frequencies [S, D].
        is_causal: Whether to use causal masking.
        scale: Attention scale factor. We use 0.0 as a sentinel value meaning
               "use default (1/sqrt(D))" because custom_op doesn't support
               Optional[float], and scale=0.0 would never be a valid real scale.
        enable_gqa: Whether to enable grouped query attention.
        rope_interleaved: Whether to use interleaved (FLUX) or NeoX (LLaMA) RoPE.

    Returns:
        Attention output [B, H, S, D] in the input dtype.
    """
    from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_rope_sdpa

    # Convert sentinel scale=0.0 back to None (meaning "use default 1/sqrt(D)").
    actual_scale = scale if scale != 0.0 else None

    return fp8_fa3_rope_sdpa(
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


@_fp8_fa3_rope_sdpa_custom_op.register_fake
def _fp8_fa3_rope_sdpa_fake(
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
    """FakeTensor implementation: tells the compiler the output shape and dtype.

    The fused kernel takes [B, S, H, D] input and produces [B, H, S, D] output
    (the transpose is baked into the kernel).
    """
    B, S, H, D = q.shape
    return torch.empty(B, H, S, D, dtype=q.dtype, device=q.device)


# Also register the non-rope fp8_fa3_sdpa as a custom op. This is used for
# SDPA nodes that don't have RoPE on their Q/K inputs. We need this because
# the fusion pass replaces ALL SDPA nodes (not just RoPE ones), and we can't
# use the monkey-patch approach (it would eat the SDPA nodes before the fusion
# pass sees them).

_NON_ROPE_CUSTOM_OP_NAME = f"{_CUSTOM_OP_LIB}::fp8_fa3_sdpa"


@torch.library.custom_op(_NON_ROPE_CUSTOM_OP_NAME, mutates_args=())
def _fp8_fa3_sdpa_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Custom op wrapper around fp8_fa3_sdpa (non-rope version).

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
    from torchao.prototype.attention.fp8_fa3.attention import fp8_fa3_sdpa

    actual_scale = scale if scale != 0.0 else None

    return fp8_fa3_sdpa(
        q,
        k,
        v,
        is_causal=is_causal,
        scale=actual_scale,
        enable_gqa=enable_gqa,
    )


@_fp8_fa3_sdpa_custom_op.register_fake
def _fp8_fa3_sdpa_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    scale: float = 0.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """FakeTensor implementation for non-rope FP8 SDPA.

    Input and output are both [B, H, S, D] (standard SDPA layout).
    The output is always contiguous even if the input is a transposed view.
    """
    return torch.empty(q.shape, dtype=q.dtype, device=q.device)


# ============================================================================
# FA3-specific Fusion Pass Entry Point
# ============================================================================


def rope_sdpa_fusion_pass(
    graph: Graph, *, fuse_rope: bool = True, strip_causal_mask: bool = False
) -> None:
    """FA3-specific fusion pass: detects and replaces SDPA patterns with FA3 ops.

    This is the entry point registered as a pre-grad custom pass via
    torch._inductor.config.pre_grad_custom_pass. It delegates to the shared
    fusion pass with FA3-specific custom ops.

    Args:
        graph: The FX graph to transform (modified in-place).
        fuse_rope: If True (default), attempt to detect and fuse RoPE
            patterns (Patterns A and B).  If False, skip RoPE detection
            and replace all fusible SDPA nodes with the non-rope FP8
            SDPA kernel only.
        strip_causal_mask: If True, the pre-flight ``detect_causal_mask``
            confirmed that every attention mask is a materialized causal
            mask, so SDPA nodes carrying a mask can have it stripped and
            replaced with ``is_causal=True``.
    """
    _shared_fusion_pass(
        graph,
        rope_sdpa_op=torch.ops.torchao.fp8_fa3_rope_sdpa.default,
        fp8_sdpa_op=torch.ops.torchao.fp8_fa3_sdpa.default,
        max_head_dim=256,
        backend_name="FA3",
        fuse_rope=fuse_rope,
        strip_causal_mask=strip_causal_mask,
    )

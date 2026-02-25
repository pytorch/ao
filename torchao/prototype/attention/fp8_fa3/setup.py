# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 FA3 backend setup: compilation, wrapping, and causal-mask pre-flight.

This module contains all FP8-FA3-specific logic for compiling a model with
the RoPE + FP8 fusion pass and wrapping it for inference.  The public entry
point is ``setup_fp8_fa3``, called by the backend-agnostic dispatcher in
``torchao.prototype.attention.api``.
"""

from functools import partial

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.attention.config import LowPrecisionAttentionConfig


class _LowPrecisionAttentionWrapper(nn.Module):
    """Opaque wrapper around a compiled low-precision attention module.

    This wrapper serves two purposes:

    1. Prevents ``torch.compile`` from tracing through the inner compiled
       module (via ``@torch._dynamo.disable`` on ``forward``), creating a
       graph-break boundary that preserves the internal compilation with
       the fusion pass.
    2. Manages FA3 activation/deactivation around each forward call so
       Dynamo traces correctly on the first (lazy-compilation) call.

    The wrapper proxies attribute access to the original (uncompiled)
    module, so model-specific attributes (e.g., ``config``) remain
    accessible.  ``_orig_mod`` is registered as a submodule so that
    ``to()``, ``cuda()``, ``eval()``, ``parameters()``, etc. propagate
    correctly through the standard ``nn.Module`` machinery.
    """

    def __init__(self, compiled_mod: nn.Module, orig_mod: nn.Module):
        super().__init__()
        # Registered as a submodule so nn.Module traversal methods
        # (parameters, to, eval, ...) reach the real weights.
        self._orig_mod = orig_mod
        # Stored outside _modules to avoid double-counting parameters
        # (the compiled module wraps the same _orig_mod).
        object.__setattr__(self, "_compiled_mod", compiled_mod)

    # ------------------------------------------------------------------
    # Attribute proxy
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # nn.Module.__getattr__ checks _parameters, _buffers, _modules.
        # If the attribute is not found there, proxy to the original module.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._orig_mod, name)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch._dynamo.disable
    def forward(self, *args, **kwargs):
        compiled_mod = object.__getattribute__(self, "_compiled_mod")
        # Activate FA3 so the first-call Dynamo trace sees the correct
        # SDPA dispatch.  On subsequent calls the compiled graph is
        # cached and FA3 activation is a harmless no-op for the fused
        # custom ops (they select the backend internally).
        activate_flash_attention_impl("FA3")
        try:
            return compiled_mod(*args, **kwargs)
        finally:
            restore_flash_attention_impl()


def setup_fp8_fa3(
    model: nn.Module,
    config: LowPrecisionAttentionConfig,
) -> nn.Module:
    """Compile *model* with the RoPE + FP8 fusion pass and wrap it."""
    if config.use_hadamard == "qkv":
        raise NotImplementedError(
            "FP8 attention with Hadamard on QKV is not yet implemented."
        )
    elif config.use_hadamard == "v":
        raise NotImplementedError(
            "FP8 attention with Hadamard on V is not yet implemented."
        )

    from torch._inductor.compile_fx import compile_fx

    from torchao.prototype.attention.fp8_fa3.fusion_pass import (
        detect_causal_mask,
        rope_sdpa_fusion_pass,
    )

    strip_causal_mask = detect_causal_mask(model)

    pass_fn = partial(
        rope_sdpa_fusion_pass,
        fuse_rope=config.fuse_rope,
        strip_causal_mask=strip_causal_mask,
    )

    def fp8_attention_backend(gm, example_inputs):
        """Custom Inductor backend that applies the RoPE + FP8 fusion pass."""
        old_pass = inductor_config.pre_grad_custom_pass
        inductor_config.pre_grad_custom_pass = pass_fn
        try:
            return compile_fx(gm, example_inputs)
        finally:
            inductor_config.pre_grad_custom_pass = old_pass

    # Clear stale Dynamo caches to ensure fresh compilation.
    torch._dynamo.reset()

    # Compile with our custom backend (fusion pass is baked in).
    compiled = torch.compile(model, backend=fp8_attention_backend)

    return _LowPrecisionAttentionWrapper(compiled, model)

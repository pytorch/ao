# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
User-facing API for low-precision attention.
"""

from functools import partial
from typing import Optional

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
import torch.nn as nn
from torch.nn.attention import (
    activate_flash_attention_impl,
    restore_flash_attention_impl,
)

from torchao.prototype.attention.config import (
    AttentionBackend,
    LowPrecisionAttentionConfig,
)
from torchao.prototype.attention.utils import (
    _check_backend_available,
    _get_available_backend,
)


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


def apply_low_precision_attention(
    model: nn.Module,
    config: Optional[LowPrecisionAttentionConfig] = None,
) -> nn.Module:
    """
    Apply low-precision attention to a model.

    Compiles the model with a custom backend that fuses
    RoPE + FP8 quantization + SDPA into optimized kernels.  The
    returned module is fully encapsulated: no global state is modified,
    and the caller does **not** need to call ``torch.compile`` or
    ``activate_flash_attention_impl`` separately.

    The returned wrapper creates a graph-break boundary, so if the
    caller later applies ``torch.compile`` to a parent model, the inner
    compilation (with the fusion pass) is preserved.

    Args:
        model: The model to apply low-precision attention to.
            Must not already be compiled with ``torch.compile``.
            For models that use KV caching (e.g., HuggingFace models
            with ``config.use_cache=True``), the caller should disable
            KV caching before calling this function.  The
            ``DynamicCache.update()`` calls insert ``torch.cat`` nodes
            that block the RoPE + SDPA fusion pass.
        config: Configuration for low-precision attention.
            If None, uses default config (auto backend selection).

    Returns:
        A wrapped module with low-precision attention compiled in.

    Raises:
        RuntimeError: If the model is already compiled or already
            wrapped.

    Example::

        from torchao.prototype.attention import apply_low_precision_attention

        model = MyTransformer()
        model = apply_low_precision_attention(model)
        output = model(inputs)  # First call triggers compilation
    """
    # Guard: already wrapped.
    if isinstance(model, _LowPrecisionAttentionWrapper):
        raise RuntimeError(
            "apply_low_precision_attention has already been applied to this module."
        )

    # Guard: already compiled.
    if isinstance(model, torch._dynamo.OptimizedModule):
        raise RuntimeError(
            "The module is already compiled with torch.compile. "
            "apply_low_precision_attention must be called on the "
            "original (uncompiled) module, before torch.compile."
        )

    if config is None:
        config = LowPrecisionAttentionConfig()

    if config.backend is None:
        backend = _get_available_backend()
    else:
        backend = config.backend
        _check_backend_available(backend)

    if backend == AttentionBackend.FP8_FA3:
        return _setup_fp8_fa3(model, config)

    raise ValueError(f"Unknown backend: {backend}")


def _setup_fp8_fa3(
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
        rope_sdpa_fusion_pass,
    )

    pass_fn = partial(rope_sdpa_fusion_pass, fuse_rope=config.fuse_rope)

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

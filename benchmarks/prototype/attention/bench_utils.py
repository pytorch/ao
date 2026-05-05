# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for attention benchmarks."""

from contextlib import contextmanager

import torch
from torch.nn.attention import SDPBackend


@contextmanager
def _set_sdpa_backend(backend: SDPBackend | None):
    """Force a specific SDPA backend without blocking the math fallback.

    Unlike ``sdpa_kernel`` which disables every backend except the chosen
    one, this disables only the *competing* accelerated backends while
    keeping math available.  This lets modules whose inputs exceed flash
    attention limits (e.g. VAE with head_dim=512) fall back to math
    automatically, and also avoids breaking ``torch.compile`` tracing
    which needs math for fake-tensor evaluation of SDPA nodes with
    ``attn_mask``.
    """
    if backend is None:
        yield
        return

    _backends = torch.backends
    prev_flash = _backends.cuda.flash_sdp_enabled()
    prev_mem = _backends.cuda.mem_efficient_sdp_enabled()
    prev_cudnn = _backends.cuda.cudnn_sdp_enabled()
    prev_math = _backends.cuda.math_sdp_enabled()

    try:
        if backend == SDPBackend.FLASH_ATTENTION:
            _backends.cuda.enable_flash_sdp(True)
            _backends.cuda.enable_cudnn_sdp(False)
            _backends.cuda.enable_mem_efficient_sdp(False)
        yield
    finally:
        _backends.cuda.enable_flash_sdp(prev_flash)
        _backends.cuda.enable_mem_efficient_sdp(prev_mem)
        _backends.cuda.enable_cudnn_sdp(prev_cudnn)
        _backends.cuda.enable_math_sdp(prev_math)

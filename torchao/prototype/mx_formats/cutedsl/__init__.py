# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Fused MXFP4 + Random Hadamard Transform CuTeDSL quantize cast.

This subpackage holds an optional, self-contained CuTeDSL implementation of an
MXFP4 (E2M1, block 32, E8M0 scales) quantize cast fused with a Random Hadamard
Transform. It is gated behind:

* a Blackwell-family GPU (SM 10.x),
* CUDA >= 12.8,
* the CuTeDSL runtime packages (``nvidia-cutlass-dsl`` and friends).

No ``cutlass`` import happens at module scope so importing this package is safe
on machines without the CuTeDSL runtime (the gate flag simply evaluates False).
"""

import torch

from torchao.utils import is_cuda_version_at_least

from .cute_utils import _cutedsl_runtime_available


def _is_sm_10x() -> bool:
    """Return True iff a Blackwell-family (SM 10.x) GPU is available."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10


_mxfp4_rht_cutedsl_kernels_available = (
    _is_sm_10x() and is_cuda_version_at_least(12, 8) and _cutedsl_runtime_available()
)


def pack32_e2m1_to_bytes(x: torch.Tensor) -> torch.Tensor:
    """Lazily re-exported test/validation entry for E2M1 packing.

    See ``cute_utils.pack32_e2m1_to_bytes``. Imported lazily so that importing
    this package does not require the CuTeDSL runtime.
    """
    from .cute_utils import pack32_e2m1_to_bytes as _impl

    return _impl(x)


def mxfp4_rht_quantize_cutedsl_2d(
    x,
    sign_vector,
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
):
    """Lazily re-exported gated wrapper for the fused MXFP4 + RHT cast.

    See ``mxfp4_rht_quantize.mxfp4_rht_quantize_cutedsl_2d``. Imported lazily so
    that importing this package does not require the CuTeDSL runtime.
    """
    from .mxfp4_rht_quantize import mxfp4_rht_quantize_cutedsl_2d as _impl

    return _impl(x, sign_vector, block_size, scaling_mode, is_swizzled_scales)


def mxfp4_rht_quantize_cutedsl(
    x,
    sign_vector,
    block_size: int = 32,
    scaling_mode: str = "floor",
    is_swizzled_scales: bool = True,
    stage_count: int = 2,
):
    """Lazily re-exported custom op for the fused MXFP4 + RHT cast.

    See ``mxfp4_rht_quantize.mxfp4_rht_quantize_cutedsl``. Imported lazily so
    that importing this package does not require the CuTeDSL runtime.
    """
    from .mxfp4_rht_quantize import mxfp4_rht_quantize_cutedsl as _impl

    return _impl(
        x, sign_vector, block_size, scaling_mode, is_swizzled_scales, stage_count
    )


def nvfp4_rht_quantize_cutedsl_2d(
    x,
    global_scale,
    sign_vector=None,
    block_size: int = 16,
    is_swizzled_scales: bool = True,
):
    """Lazily re-exported gated wrapper for the fused NVFP4 (+/- RHT) cast.

    See ``nvfp4_rht_quantize.nvfp4_rht_quantize_cutedsl_2d``. Imported lazily so
    that importing this package does not require the CuTeDSL runtime.
    ``sign_vector=None`` (or empty) selects the plain NVFP4 cast.
    """
    from .nvfp4_rht_quantize import nvfp4_rht_quantize_cutedsl_2d as _impl

    return _impl(x, global_scale, sign_vector, block_size, is_swizzled_scales)


def nvfp4_rht_quantize_cutedsl(
    x,
    global_scale,
    sign_vector,
    block_size: int = 16,
    is_swizzled_scales: bool = True,
    stage_count: int = 2,
):
    """Lazily re-exported custom op for the fused NVFP4 (+/- RHT) cast.

    See ``nvfp4_rht_quantize.nvfp4_rht_quantize_cutedsl``. Imported lazily so
    that importing this package does not require the CuTeDSL runtime.
    """
    from .nvfp4_rht_quantize import nvfp4_rht_quantize_cutedsl as _impl

    return _impl(
        x, global_scale, sign_vector, block_size, is_swizzled_scales, stage_count
    )


__all__ = [
    "_is_sm_10x",
    "_mxfp4_rht_cutedsl_kernels_available",
    "pack32_e2m1_to_bytes",
    "mxfp4_rht_quantize_cutedsl_2d",
    "mxfp4_rht_quantize_cutedsl",
    "nvfp4_rht_quantize_cutedsl_2d",
    "nvfp4_rht_quantize_cutedsl",
]

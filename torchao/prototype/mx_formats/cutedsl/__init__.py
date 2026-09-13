# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unified FP4 (NVFP4 + MXFP4) +/- Random Hadamard Transform CuTeDSL casts.

A single no-smem streaming CuTeDSL quantize kernel for both FP4 formats
(NVFP4 E2M1 block-16 with a two-level E4M3 scale; MXFP4 E2M1 block-32 with an
E8M0 scale), all three GEMM scale layouts (``linear`` / ``cublas_blocked`` /
``mma_tiled``), and an optional fused Random Hadamard Transform. Gated behind:

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


_fp4_cutedsl_kernels_available = (
    _is_sm_10x() and is_cuda_version_at_least(12, 8) and _cutedsl_runtime_available()
)


def pack32_e2m1_to_bytes(x: torch.Tensor) -> torch.Tensor:
    """Lazily re-exported test/validation entry for E2M1 packing.

    See ``cute_utils.pack32_e2m1_to_bytes``. Imported lazily so that importing
    this package does not require the CuTeDSL runtime.
    """
    from .cute_utils import pack32_e2m1_to_bytes as _impl

    return _impl(x)


def fp4_quantize_unified_2d(
    x,
    sign_vector=None,
    fmt: str = "nvfp4",
    scaling_mode: str = "floor",
    scale_layout: str = "cublas_blocked",
):
    """Lazily re-exported gated wrapper for the unified FP4 (+/- RHT) cast.

    See ``fp4_unified_quantize.fp4_quantize_unified_2d``. Imported lazily so
    that importing this package does not require the CuTeDSL runtime. One kernel
    serves NVFP4 (``fmt="nvfp4"``) and MXFP4 (``fmt="mxfp4"``) across the
    ``linear`` / ``cublas_blocked`` / ``mma_tiled`` scale layouts; an empty /
    ``None`` ``sign_vector`` selects the plain cast.
    """
    from .fp4_unified_quantize import fp4_quantize_unified_2d as _impl

    return _impl(x, sign_vector, fmt, scaling_mode, scale_layout)


__all__ = [
    "_is_sm_10x",
    "_fp4_cutedsl_kernels_available",
    "pack32_e2m1_to_bytes",
    "fp4_quantize_unified_2d",
]

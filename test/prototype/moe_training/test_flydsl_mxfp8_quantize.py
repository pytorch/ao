# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the FlyDSL-based MXFP8 quantization kernels (AMD GPUs).

These mirror the CuteDSL tests in ``test_kernels.py`` but cover the AMD
counterpart implementation. Skipped on CUDA hardware and when FlyDSL is
not installed.
"""

import pytest
import torch

from torchao.utils import is_MI300, is_MI350

# Hardware gate: FlyDSL kernels target AMD CDNA3+ / RDNA4+ via ROCm/HIP.
if not (torch.cuda.is_available() and (is_MI300() or is_MI350())):
    pytest.skip(
        "FlyDSL MXFP8 kernels require an MI300 or MI350-class AMD GPU",
        allow_module_level=True,
    )

# FlyDSL-availability gate: skip the whole module if the package can't be imported.
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_utils import (
    _flydsl_runtime_available,
    _missing_flydsl_runtime_packages,
)

if not _flydsl_runtime_available():
    pytest.skip(
        f"FlyDSL is not available (missing: {', '.join(_missing_flydsl_runtime_packages())}). "
        "Install FlyDSL and put its python_packages dir on PYTHONPATH.",
        allow_module_level=True,
    )

from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_2d_1x32 import (
    mxfp8_quantize_flydsl_2d_1x32,
)


# Common K values that satisfy the baseline kernel's K % 2048 == 0 constraint.
# (Tail handling is a planned follow-up; once implemented, smaller Ks here.)
_VALID_K = (2048, 4096, 8192)
_VALID_M = (1, 32, 128, 1024)


@pytest.mark.parametrize("M", _VALID_M)
@pytest.mark.parametrize("K", _VALID_K)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
def test_flydsl_quantize_2d_1x32_floor_numerics(M, K, input_dtype):
    """Bit-exact match of FlyDSL FLOOR quantize vs torchao to_mx reference."""
    torch.manual_seed(0)
    # Mix of magnitudes per row so per-block amax varies; *30 makes occasional
    # clamps to ±448 once scaled (exercises FLOOR-mode saturation behavior).
    x = (torch.randn(M, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    # FlyDSL kernel.
    q_fly, s_fly = mxfp8_quantize_flydsl_2d_1x32(
        x, block_size=32, scaling_mode="floor",
    )

    # torchao to_mx reference (FLOOR mode).
    s_ref, q_ref = to_mx(
        x,
        elem_dtype=torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode.FLOOR,
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(M, K)
    s_ref_u8 = s_ref.view(M, K // 32)

    # Bit-exact (byte-equal) compare for both data and scales.
    torch.testing.assert_close(
        q_fly.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        s_fly.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    # Layout / shape sanity.
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.shape == (M, K)
    assert q_fly.stride() == (K, 1), "q_data must be row-major"
    assert s_fly.dtype == torch.float8_e8m0fnu
    assert s_fly.shape == (M, K // 32)


def test_flydsl_quantize_2d_1x32_rejects_unsupported_dtype():
    x = torch.randn(64, 2048, dtype=torch.float16, device="cuda")
    with pytest.raises(AssertionError, match="bfloat16 or float32"):
        mxfp8_quantize_flydsl_2d_1x32(x)


def test_flydsl_quantize_2d_1x32_rejects_unsupported_block_size():
    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="block_size=32"):
        mxfp8_quantize_flydsl_2d_1x32(x, block_size=64)


def test_flydsl_quantize_2d_1x32_rejects_K_not_multiple_of_2048():
    # 1024 is divisible by 32 (block size) but NOT by 2048 (chunk size).
    x = torch.randn(64, 1024, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="multiple of 2048"):
        mxfp8_quantize_flydsl_2d_1x32(x)


def test_flydsl_quantize_2d_1x32_rejects_rceil_for_now():
    """RCEIL is a planned follow-up — should raise a clear NotImplementedError."""
    x = torch.randn(64, 2048, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="RCEIL"):
        mxfp8_quantize_flydsl_2d_1x32(x, scaling_mode="rceil")

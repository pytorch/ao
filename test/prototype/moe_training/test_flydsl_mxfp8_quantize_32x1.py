# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the FlyDSL MXFP8 2D 32x1 quantization kernel (AMD GPUs).

Mirrors test_flydsl_mxfp8_quantize.py but for the M-direction (32x1) kernel.
"""

import pytest
import torch

from torchao.utils import is_MI300, is_MI350

if not (torch.cuda.is_available() and (is_MI300() or is_MI350())):
    pytest.skip(
        "FlyDSL MXFP8 32x1 kernel requires an MI300 or MI350-class AMD GPU",
        allow_module_level=True,
    )

from torchao.prototype.moe_training.kernels.mxfp8.flydsl_utils import (
    _flydsl_runtime_available,
    _missing_flydsl_runtime_packages,
)

if not _flydsl_runtime_available():
    pytest.skip(
        f"FlyDSL not available (missing: {', '.join(_missing_flydsl_runtime_packages())}).",
        allow_module_level=True,
    )

from torchao.prototype.mx_formats.mx_tensor import ScaleCalculationMode, to_mx
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_2d_32x1 import (
    mxfp8_quantize_flydsl_2d_32x1,
)


# Constraints: M % 32 == 0 (block size), K % 64 == 0 (wave-aligned tile).
_VALID_M = (32, 64, 128, 1024)
_VALID_K = (64, 128, 2048, 4096)


@pytest.mark.parametrize("M", _VALID_M)
@pytest.mark.parametrize("K", _VALID_K)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
def test_flydsl_quantize_2d_32x1_floor_numerics(M, K, input_dtype):
    """Bit-exact match of FlyDSL FLOOR 32x1 vs torchao to_mx reference."""
    torch.manual_seed(0)
    x = (torch.randn(M, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    q_fly, s_fly = mxfp8_quantize_flydsl_2d_32x1(
        x, block_size=32, scaling_mode="floor",
    )

    # Reference: quantize x.T (K, M) along its last dim (= M-direction of x).
    x_t = x.transpose(0, 1).contiguous()
    s_ref, q_ref = to_mx(
        x_t, elem_dtype=torch.float8_e4m3fn,
        block_size=32, scaling_mode=ScaleCalculationMode.FLOOR,
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(K, M)
    s_ref_u8 = s_ref.view(K, M // 32)

    # q_fly is col-major (M, K) with stride (1, M); q_ref_fp8 is row-major (K, M).
    # They represent the same data: q_fly[m, k] == q_ref_fp8[k, m].
    # Compare element-wise via the transpose.
    fly_t = q_fly.transpose(0, 1).contiguous()
    torch.testing.assert_close(
        fly_t.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s_fly.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0, atol=0,
    )

    # Layout / shape sanity.
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.shape == (M, K)
    assert q_fly.stride() == (1, M), "q_data must be col-major"
    assert s_fly.dtype == torch.float8_e8m0fnu
    assert s_fly.shape == (K, M // 32)


def test_flydsl_quantize_2d_32x1_rejects_unsupported_dtype():
    x = torch.randn(64, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(AssertionError, match="bfloat16 or float32"):
        mxfp8_quantize_flydsl_2d_32x1(x)


def test_flydsl_quantize_2d_32x1_rejects_unsupported_block_size():
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="block_size=32"):
        mxfp8_quantize_flydsl_2d_32x1(x, block_size=64)


def test_flydsl_quantize_2d_32x1_rejects_M_not_multiple_of_32():
    # 16 isn't a multiple of 32.
    x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="divisible by block_size"):
        mxfp8_quantize_flydsl_2d_32x1(x)


def test_flydsl_quantize_2d_32x1_rejects_K_not_multiple_of_64():
    x = torch.randn(64, 96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="divisible by"):
        mxfp8_quantize_flydsl_2d_32x1(x)


def test_flydsl_quantize_2d_32x1_rejects_rceil_for_now():
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="RCEIL"):
        mxfp8_quantize_flydsl_2d_32x1(x, scaling_mode="rceil")

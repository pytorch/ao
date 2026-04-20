# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the FlyDSL MXFP8 3D MoE quantization kernel (AMD GPUs)."""

import pytest
import torch

from torchao.utils import is_MI300, is_MI350

if not (torch.cuda.is_available() and (is_MI300() or is_MI350())):
    pytest.skip(
        "FlyDSL MXFP8 3D kernel requires an MI300 or MI350-class AMD GPU",
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
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_3d import (
    mxfp8_quantize_flydsl_3d,
)


_VALID_E = (1, 2, 4, 8)
_VALID_N = (32, 64, 256)
_VALID_K = (64, 128, 4096)


@pytest.mark.parametrize("E", _VALID_E)
@pytest.mark.parametrize("N", _VALID_N)
@pytest.mark.parametrize("K", _VALID_K)
@pytest.mark.parametrize("input_dtype", (torch.bfloat16, torch.float32))
def test_flydsl_quantize_3d_floor_numerics(E, N, K, input_dtype):
    """Bit-exact match of FlyDSL FLOOR 3D quantize vs torchao to_mx reference."""
    torch.manual_seed(0)
    x = (torch.randn(E, N, K, dtype=input_dtype, device="cuda") * 30.0).contiguous()

    q_fly, s_fly = mxfp8_quantize_flydsl_3d(
        x, block_size=32, scaling_mode="floor",
    )

    # Reference: to_mx along last dim of x.transpose(1, 2) = (E, K, N).
    x_t = x.transpose(1, 2).contiguous()
    s_ref, q_ref = to_mx(
        x_t, elem_dtype=torch.float8_e4m3fn,
        block_size=32, scaling_mode=ScaleCalculationMode.FLOOR,
    )
    q_ref_fp8 = q_ref.to(torch.float8_e4m3fn).view(E, K, N)
    s_ref_u8 = s_ref.view(E, K, N // 32)

    # q_fly is (E, N, K) per-expert col-major; q_ref_fp8 is (E, K, N) row-major.
    # q_fly[e, n, k] == q_ref_fp8[e, k, n] when the kernel is correct.
    fly_t = q_fly.transpose(1, 2).contiguous()    # (E, K, N) row-major
    torch.testing.assert_close(
        fly_t.view(torch.uint8),
        q_ref_fp8.view(torch.uint8),
        rtol=0, atol=0,
    )

    # s_fly is (E, N//32, K); s_ref is (E, K, N//32). Transpose s_fly to compare.
    s_fly_t = s_fly.transpose(1, 2).contiguous()
    torch.testing.assert_close(
        s_fly_t.view(torch.uint8),
        s_ref_u8.view(torch.uint8),
        rtol=0, atol=0,
    )

    # Layout / shape sanity.
    assert q_fly.dtype == torch.float8_e4m3fn
    assert q_fly.shape == (E, N, K)
    assert q_fly.stride() == (N * K, 1, N), "q_data must be per-expert col-major"
    assert s_fly.dtype == torch.float8_e8m0fnu
    assert s_fly.shape == (E, N // 32, K)


def test_flydsl_quantize_3d_rejects_unsupported_dtype():
    x = torch.randn(2, 64, 128, dtype=torch.float16, device="cuda")
    with pytest.raises(AssertionError, match="bfloat16 or float32"):
        mxfp8_quantize_flydsl_3d(x)


def test_flydsl_quantize_3d_rejects_unsupported_block_size():
    x = torch.randn(2, 64, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="block_size=32"):
        mxfp8_quantize_flydsl_3d(x, block_size=64)


def test_flydsl_quantize_3d_rejects_N_not_multiple_of_32():
    x = torch.randn(2, 16, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="divisible by block_size"):
        mxfp8_quantize_flydsl_3d(x)


def test_flydsl_quantize_3d_rejects_K_not_multiple_of_64():
    x = torch.randn(2, 32, 96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="divisible by"):
        mxfp8_quantize_flydsl_3d(x)


def test_flydsl_quantize_3d_rejects_2d_input():
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="3D input"):
        mxfp8_quantize_flydsl_3d(x)


def test_flydsl_quantize_3d_rejects_rceil_for_now():
    x = torch.randn(2, 32, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="RCEIL"):
        mxfp8_quantize_flydsl_3d(x, scaling_mode="rceil")

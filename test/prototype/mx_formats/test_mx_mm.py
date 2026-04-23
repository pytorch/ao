# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial

import pytest
import torch
import torch.nn.functional as F

from torchao.float8.float8_utils import compute_error
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.mx_formats.utils import to_blocked
from torchao.utils import (
    is_sm_at_least_100,
    torch_version_at_least,
)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

# ScalingType and SwizzleType are only available in PyTorch 2.10+
if torch_version_at_least("2.10.0"):
    from torch.nn.functional import ScalingType, SwizzleType


def run_matrix_test(M: int, K: int, N: int, format) -> float:
    dtype = torch.bfloat16
    device = torch.accelerator.current_accelerator()

    a = torch.rand((M, K), dtype=dtype, device=device)
    b = torch.rand((N, K), dtype=dtype, device=device)

    fmt = torch.float8_e4m3fn if format == "fp8" else torch.float4_e2m1fn_x2

    a_mx = MXTensor.to_mx(a, fmt, 32)
    b_mx = MXTensor.to_mx(b, fmt, 32)

    a_data = a_mx.qdata
    b_data = b_mx.qdata
    assert b_data.is_contiguous()
    b_data = b_data.transpose(-1, -2)

    a_scale = a_mx.scale.view(M, K // 32)
    b_scale = b_mx.scale.view(N, K // 32)

    if device == torch.device("cuda"):
        a_scale_block = to_blocked(a_scale)
        b_scale_block = to_blocked(b_scale)
    else:
        a_scale_block = a_scale
        b_scale_block = b_scale

    if not torch_version_at_least("2.10.0"):
        mx_func = partial(
            torch._scaled_mm,
            scale_a=a_scale_block,
            scale_b=b_scale_block,
            out_dtype=torch.bfloat16,
        )
    else:
        swizzle_type = (
            SwizzleType.SWIZZLE_32_4_4
            if device == torch.device("cuda")
            else SwizzleType.NO_SWIZZLE
        )
        mx_func = partial(
            F.scaled_mm,
            scale_a=a_scale_block,
            scale_recipe_a=ScalingType.BlockWise1x32,
            scale_b=b_scale_block,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_a=swizzle_type,
            swizzle_b=swizzle_type,
            output_dtype=torch.bfloat16,
        )

    out_hp = a_mx.dequantize(torch.bfloat16) @ b_mx.dequantize(
        torch.bfloat16
    ).transpose(-1, -2)
    out = mx_func(a_data.view(fmt), b_data.view(fmt))

    return compute_error(out_hp, out).item()


@pytest.mark.skipif(
    not torch.accelerator.is_available(), reason="No accelerator available"
)
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm_at_least_100(),
    reason="CUDA capability >= 10.0 required for mxfloat8",
)
@pytest.mark.skipif(
    torch.xpu.is_available() and not torch_version_at_least("2.13.0"),
    reason="XPU blockwise scaled_mm requires oneDNN 3.11+ (PyTorch 2.13+) (pytorch/pytorch#173630, #176043, #176064)",
)
@pytest.mark.parametrize(
    "size",
    [
        (128, 128, 128),
        (256, 256, 256),
        (384, 384, 384),  # Small
        (512, 512, 512),
        (768, 768, 768),  # Medium
        (1024, 1024, 1024),
        (8192, 8192, 8192),  # Large
        (128, 256, 384),
        (256, 384, 512),  # Non-square
        (129, 256, 384),
        (133, 512, 528),  # Non-aligned
    ],
    ids=lambda x: f"{x[0]}x{x[1]}x{x[2]}",
)
@pytest.mark.parametrize(
    "format", ["fp8", "fp4"] if torch_version_at_least("2.10.0") else ["fp8"]
)
def test_matrix_multiplication(size, format):
    M, K, N = size
    sqnr = run_matrix_test(M, K, N, format)
    threshold = 80.0
    assert sqnr >= threshold, (
        f"{format} SQNR {sqnr} below threshold for dims {M}x{K}x{N}"
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the unified FP4 (NVFP4 + MXFP4) +/- RHT CuTeDSL quantize cast.

Validates that the plain (no-RHT) cast's per-block scales are byte-exact vs the
``cute_utils`` host scale references (themselves bit-exact vs eager torchao),
that the two thread mappings (striped / warp-per-row) produce identical output,
that qdata is invariant across the three scale layouts, and that the plain cast
round-trips to within FP4 error.
"""

import pytest
import torch

from torchao.prototype.mx_formats.cutedsl import _fp4_cutedsl_kernels_available

pytestmark = pytest.mark.skipif(
    not _fp4_cutedsl_kernels_available,
    reason="requires SM 10.x (Blackwell), CUDA>=12.8, and the CuTeDSL runtime",
)

_E2M1_VALS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _x(M, K, dtype=torch.bfloat16):
    torch.manual_seed(0)
    return (torch.randn(M, K, device="cuda", dtype=dtype) * 5).contiguous()


def _dequant_linear(qdata, scales_u8, gs, M, K, fmt):
    blk = 16 if fmt == "nvfp4" else 32
    kb = K // blk
    lut = torch.tensor(_E2M1_VALS + [-v for v in _E2M1_VALS], device=qdata.device)
    codes = torch.stack(
        [lut[(qdata & 0xF).long()], lut[(qdata >> 4).long()]], dim=-1
    ).reshape(M, K)
    if fmt == "nvfp4":
        sc = scales_u8.view(torch.float8_e4m3fn).float().view(M, kb) / gs
    else:
        sc = scales_u8.view(torch.float8_e8m0fnu).float().view(M, kb)
    return (codes.view(M, kb, blk) * sc.unsqueeze(-1)).view(M, K)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("M,K", [(256, 512), (2304, 4096), (1024, 2048)])
def test_nvfp4_plain_scales_match_reference(dtype, M, K):
    # Unified linear-layout E4M3 scales must be byte-exact vs the cute_utils
    # per-block host reference (bit-exact vs eager NVFP4).
    from torchao.prototype.mx_formats.cutedsl.cute_utils import (
        compute_block_scale_e4m3_nvfp4,
    )
    from torchao.prototype.mx_formats.cutedsl.fp4_unified_quantize import (
        _fp4_quantize_unified_impl,
    )

    x = _x(M, K, dtype)
    gs = 2688.0 / x.abs().max().item()
    ref = compute_block_scale_e4m3_nvfp4(x, gs)  # (M, K//16) float8_e4m3fn
    _, su = _fp4_quantize_unified_impl(
        x, fmt="nvfp4", scale_layout="linear", global_scale=gs, mapping="striped"
    )
    torch.cuda.synchronize()
    assert int((ref.view(torch.uint8).view(-1) != su.view(-1)).sum()) == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("M,K", [(256, 512), (2304, 4096), (1024, 2048)])
@pytest.mark.parametrize("mode", ["floor", "rceil"])
def test_mxfp4_plain_scales_match_reference(dtype, M, K, mode):
    from torchao.prototype.mx_formats.cutedsl.cute_utils import (
        compute_block_scale_e8m0_fp4,
    )
    from torchao.prototype.mx_formats.cutedsl.fp4_unified_quantize import (
        _fp4_quantize_unified_impl,
    )

    x = _x(M, K, dtype)
    ref = compute_block_scale_e8m0_fp4(x, mode)  # (M, K//32) float8_e8m0fnu
    _, su = _fp4_quantize_unified_impl(
        x, fmt="mxfp4", scaling_mode=mode, scale_layout="linear", mapping="striped"
    )
    torch.cuda.synchronize()
    assert int((ref.view(torch.uint8).view(-1) != su.view(-1)).sum()) == 0


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
@pytest.mark.parametrize("rht", [False, True])
@pytest.mark.parametrize("scale_layout", ["linear", "cublas_blocked", "mma_tiled"])
def test_wpr_matches_striped(fmt, rht, scale_layout):
    # The two mappings are a pure work-distribution change: identical output.
    from torchao.prototype.mx_formats.cutedsl.fp4_unified_quantize import (
        _fp4_quantize_unified_impl,
    )

    x = _x(256, 2048)
    gs = 2688.0 / x.abs().max().item() if fmt == "nvfp4" else 1.0
    sign = ([1, -1] * (8 if fmt == "nvfp4" else 16)) if rht else None
    qs, ss = _fp4_quantize_unified_impl(
        x, sign_vector=sign, fmt=fmt, scale_layout=scale_layout, global_scale=gs,
        mapping="striped",
    )
    qw, sw = _fp4_quantize_unified_impl(
        x, sign_vector=sign, fmt=fmt, scale_layout=scale_layout, global_scale=gs,
        mapping="wpr", warps=4, xsplit=2, ilp=2,
    )
    torch.cuda.synchronize()
    assert int((qs != qw).sum()) == 0
    assert int((ss.view(-1) != sw.view(-1)).sum()) == 0


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_qdata_layout_invariant(fmt):
    from torchao.prototype.mx_formats.cutedsl.fp4_unified_quantize import (
        _fp4_quantize_unified_impl,
    )

    x = _x(256, 2048)
    gs = 2688.0 / x.abs().max().item() if fmt == "nvfp4" else 1.0
    q = {}
    for lay in ("linear", "cublas_blocked", "mma_tiled"):
        q[lay], _ = _fp4_quantize_unified_impl(
            x, fmt=fmt, scale_layout=lay, global_scale=gs, mapping="striped"
        )
    torch.cuda.synchronize()
    assert int((q["linear"] != q["cublas_blocked"]).sum()) == 0
    assert int((q["linear"] != q["mma_tiled"]).sum()) == 0


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp4"])
def test_dequant_roundtrip_plain(fmt):
    from torchao.prototype.mx_formats.cutedsl.fp4_unified_quantize import (
        _fp4_quantize_unified_impl,
    )

    x = _x(256, 2048)
    gs = 2688.0 / x.abs().max().item() if fmt == "nvfp4" else 1.0
    q, s = _fp4_quantize_unified_impl(
        x, fmt=fmt, scale_layout="linear", global_scale=gs, mapping="striped"
    )
    torch.cuda.synchronize()
    deq = _dequant_linear(q, s, gs, 256, 2048, fmt)
    xf = x.float()
    m = xf.abs() > 0.1
    rel = ((deq - xf).abs()[m] / xf.abs()[m]).median().item()
    assert rel < 0.3


def test_custom_op_smoke():
    from torchao.prototype.mx_formats.cutedsl import fp4_quantize_unified_2d

    q, s = fp4_quantize_unified_2d(_x(128, 2048), None, "nvfp4", "floor", "mma_tiled")
    assert q.shape == (128, 1024)

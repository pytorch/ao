# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the public FlyDSL MXFP8 dispatcher API in
``torchao.prototype.moe_training.kernels.mxfp8.quant`` — verifies that the
``torchao::mxfp8_quantize_*_flydsl`` custom-ops are registered and that the
public Python wrappers produce the same output as calling the underlying
FlyDSL kernels directly.
"""

import pytest
import torch

from torchao.utils import is_MI300, is_MI350

if not (torch.cuda.is_available() and (is_MI300() or is_MI350())):
    pytest.skip(
        "FlyDSL MXFP8 dispatcher requires an MI300 or MI350-class AMD GPU",
        allow_module_level=True,
    )

from torchao.prototype.moe_training.kernels.mxfp8 import (
    _mxfp8_flydsl_kernels_available,
    mxfp8_quantize_2d_1x32_flydsl,
    mxfp8_quantize_2d_32x1_flydsl,
    mxfp8_quantize_3d_flydsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_2d_1x32 import (
    mxfp8_quantize_flydsl_2d_1x32,
)
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_2d_32x1 import (
    mxfp8_quantize_flydsl_2d_32x1,
)
from torchao.prototype.moe_training.kernels.mxfp8.flydsl_quantize_3d import (
    mxfp8_quantize_flydsl_3d,
)


if not _mxfp8_flydsl_kernels_available:
    pytest.skip(
        "FlyDSL kernels unavailable (FlyDSL not installed?)",
        allow_module_level=True,
    )


def test_flydsl_custom_ops_registered():
    """The three torchao::mxfp8_quantize_*_flydsl custom_ops exist."""
    assert hasattr(torch.ops.torchao, "mxfp8_quantize_2d_1x32_flydsl")
    assert hasattr(torch.ops.torchao, "mxfp8_quantize_2d_32x1_flydsl")
    assert hasattr(torch.ops.torchao, "mxfp8_quantize_3d_flydsl")


def test_dispatcher_2d_1x32_matches_direct_call():
    """Calling through the public dispatcher equals the direct kernel call."""
    torch.manual_seed(0)
    x = (torch.randn(64, 4096, dtype=torch.bfloat16, device="cuda") * 30.0).contiguous()

    q_disp, s_disp = mxfp8_quantize_2d_1x32_flydsl(x)
    q_direct, s_direct = mxfp8_quantize_flydsl_2d_1x32(x)

    torch.testing.assert_close(
        q_disp.view(torch.uint8), q_direct.view(torch.uint8), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s_disp.view(torch.uint8), s_direct.view(torch.uint8), rtol=0, atol=0,
    )
    assert q_disp.dtype == torch.float8_e4m3fn
    assert s_disp.dtype == torch.float8_e8m0fnu


def test_dispatcher_2d_32x1_matches_direct_call():
    torch.manual_seed(0)
    x = (torch.randn(64, 128, dtype=torch.bfloat16, device="cuda") * 30.0).contiguous()

    q_disp, s_disp = mxfp8_quantize_2d_32x1_flydsl(x)
    q_direct, s_direct = mxfp8_quantize_flydsl_2d_32x1(x)

    # Same col-major layout — compare per-element via the transpose like the
    # 32x1 numerics test does, since flat byte order of col-major tensors
    # depends on .flatten()'s traversal order.
    q_disp_t = q_disp.transpose(0, 1).contiguous()
    q_direct_t = q_direct.transpose(0, 1).contiguous()
    torch.testing.assert_close(
        q_disp_t.view(torch.uint8), q_direct_t.view(torch.uint8), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s_disp.view(torch.uint8), s_direct.view(torch.uint8), rtol=0, atol=0,
    )
    assert q_disp.stride() == (1, 64), "dispatcher must preserve col-major layout"


def test_dispatcher_3d_matches_direct_call():
    torch.manual_seed(0)
    x = (torch.randn(4, 64, 256, dtype=torch.bfloat16, device="cuda") * 30.0).contiguous()

    q_disp, s_disp = mxfp8_quantize_3d_flydsl(x)
    q_direct, s_direct = mxfp8_quantize_flydsl_3d(x)

    q_disp_t = q_disp.transpose(1, 2).contiguous()
    q_direct_t = q_direct.transpose(1, 2).contiguous()
    torch.testing.assert_close(
        q_disp_t.view(torch.uint8), q_direct_t.view(torch.uint8), rtol=0, atol=0,
    )
    torch.testing.assert_close(
        s_disp.view(torch.uint8), s_direct.view(torch.uint8), rtol=0, atol=0,
    )
    E, N, K = x.shape
    assert q_disp.stride() == (N * K, 1, N), "dispatcher must preserve per-expert col-major"

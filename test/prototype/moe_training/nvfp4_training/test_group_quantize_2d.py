# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the CuteDSL per-expert (grouped) 2D NVFP4 weight quantize (SM100+).

The grouped op invokes the dense no-RHT kernel per expert, so the primary contract
is bitwise equality with ``cutedsl_weight_quantize_2d`` on each ``A[e]`` slice.
Quantization quality of the dense kernel itself is covered by test_quantize_2d.py.
"""

import pytest
import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_training.group_quantize_2d_cutedsl import (
    cutedsl_group_weight_quantize_2d,
)
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.nvfp4_training.quantize_2d_cutedsl import (
    cutedsl_weight_quantize_2d,
)
from torchao.prototype.mx_formats.nvfp4_tensor import (
    NVFP4Tensor,
    per_tensor_amax_to_scale,
)

_skip_no_cutedsl = pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(),
    reason="requires SM100 (Blackwell) + CuteDSL runtime (cuda-python, nvidia-cutlass-dsl)",
)

# (E, M, N): 256-row and 128-row supertile shapes.
_SHAPES = [
    pytest.param(3, 256, 256, id="E3_M256_N256"),
    pytest.param(2, 1408, 512, id="E2_M1408_N512"),
    pytest.param(4, 128, 128, id="E4_M128_N128"),
    pytest.param(1, 512, 384, id="E1_M512_N384"),
]


def _dequantize(codes, sf, amax):
    return (
        NVFP4Tensor(
            codes,
            sf,
            16,
            torch.bfloat16,
            per_tensor_scale=per_tensor_amax_to_scale(amax.reshape(())),
            is_swizzled_scales=True,
        )
        .dequantize()
        .float()
    )


@_skip_no_cutedsl
@pytest.mark.parametrize("E,M,N", _SHAPES)
@torch.no_grad()
def test_group_weight_quantize_matches_dense_per_expert(E, M, N):
    """Every expert slice of the grouped op is bitwise equal to the dense op on A[e]."""
    torch.manual_seed(0)
    A = torch.randn(E, M, N, dtype=torch.bfloat16, device="cuda") * (
        torch.arange(1, E + 1, device="cuda").view(E, 1, 1)
    )
    amax = A.float().abs().amax(dim=(1, 2)).contiguous()

    g_codes, g_sf, g_t_codes, g_t_sf = cutedsl_group_weight_quantize_2d(A, amax, E)

    assert g_codes.shape == (E, M, N // 2) and g_codes.dtype == torch.uint8
    assert g_sf.shape == (E, M // 128, N // 64, 32, 16)
    assert g_t_codes.shape == (E, N, M // 2) and g_t_codes.dtype == torch.uint8
    assert g_t_sf.shape == (E, N // 128, M // 64, 32, 16)

    for e in range(E):
        d_codes, d_sf, d_t_codes, d_t_sf = cutedsl_weight_quantize_2d(
            A[e].contiguous(), amax[e].reshape(1)
        )
        torch.testing.assert_close(g_codes[e], d_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            g_sf[e].view(torch.uint8), d_sf.view(torch.uint8), atol=0, rtol=0
        )
        torch.testing.assert_close(g_t_codes[e], d_t_codes, atol=0, rtol=0)
        torch.testing.assert_close(
            g_t_sf[e].view(torch.uint8), d_t_sf.view(torch.uint8), atol=0, rtol=0
        )


@_skip_no_cutedsl
@torch.no_grad()
def test_group_weight_quantize_sqnr():
    """Dequantized W and W.T reconstruct each expert with SQNR >= 18 dB (the dense
    suite's cutedsl threshold for 2D 16x16 block scaling of gaussian data)."""
    torch.manual_seed(1)
    E, M, N = 2, 256, 512
    A = torch.randn(E, M, N, dtype=torch.bfloat16, device="cuda")
    amax = A.float().abs().amax(dim=(1, 2)).contiguous()
    codes, sf, t_codes, t_sf = cutedsl_group_weight_quantize_2d(A, amax, E)
    for e in range(E):
        w = _dequantize(codes[e], sf[e], amax[e])
        wt = _dequantize(t_codes[e], t_sf[e], amax[e])
        assert compute_error(A[e].float(), w) >= 18.0
        assert compute_error(A[e].t().float().contiguous(), wt) >= 18.0


@_skip_no_cutedsl
@torch.no_grad()
def test_group_weight_quantize_invalid_inputs_raise():
    A = torch.randn(2, 256, 256, dtype=torch.bfloat16, device="cuda")
    amax = A.float().abs().amax(dim=(1, 2)).contiguous()
    with pytest.raises(ValueError, match="experts"):
        cutedsl_group_weight_quantize_2d(A, amax, 3)
    with pytest.raises(ValueError, match="3-D"):
        cutedsl_group_weight_quantize_2d(A[0], amax[:1], 1)
    with pytest.raises(ValueError):
        cutedsl_group_weight_quantize_2d(
            torch.randn(2, 192, 256, dtype=torch.bfloat16, device="cuda"), amax, 2
        )
    with pytest.raises(ValueError):
        cutedsl_group_weight_quantize_2d(A, amax.to(torch.float64), 2)


@_skip_no_cutedsl
@torch.no_grad()
def test_group_weight_quantize_fake_shapes():
    """register_fake propagates the E-batched output shapes (torch.compile support)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        A = torch.empty(3, 384, 256, dtype=torch.bfloat16, device="cuda")
        amax = torch.empty(3, dtype=torch.float32, device="cuda")
        codes, sf, t_codes, t_sf = cutedsl_group_weight_quantize_2d(A, amax, 3)
    assert codes.shape == (3, 384, 128)
    assert sf.shape == (3, 3, 4, 32, 16)
    assert t_codes.shape == (3, 256, 192)
    assert t_sf.shape == (3, 2, 6, 32, 16)

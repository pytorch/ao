# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90, torch_version_at_least

if not (
    torch_version_at_least("2.7.0")
    and torch.cuda.is_available()
    and (is_sm_at_least_90() or is_MI300() or is_MI350())
):
    pytest.skip(
        "Requires SM90+ GPU (torch._grouped_mm), MI300, or MI350",
        allow_module_level=True,
    )

from torchao.float8.float8_utils import compute_error
from torchao.prototype.moe_training.nvfp4_grouped_mm import (
    _emulated_nvfp4_scaled_grouped_mm_2d_2d,
    _emulated_nvfp4_scaled_grouped_mm_2d_3d,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
from torchao.testing.utils import skip_if_rocm

BLOCK_SIZE = 16


def _quantize_for_test(x: torch.Tensor):
    """Quantize a tensor using nvfp4_quantize and return (packed_data, scales)."""
    scales, packed_data = nvfp4_quantize(x, block_size=BLOCK_SIZE)
    return packed_data, scales


def _quantize_3d_for_test(w: torch.Tensor):
    """Quantize a 3D expert weight tensor (E, N, K) per-expert.

    Returns (packed_data, scales) with shapes:
        packed_data: (E, N, K//2)
        scales: (E, N, K//block_size)
    """
    packed_list, scales_list = [], []
    for i in range(w.shape[0]):
        packed, scales = _quantize_for_test(w[i].contiguous())
        packed_list.append(packed)
        scales_list.append(scales)
    return torch.stack(packed_list), torch.stack(scales_list)


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_emulated_nvfp4_grouped_gemm_2d_3d(M, K, N, num_experts):
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w_t = torch.randn(num_experts, K, N, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M)
    x_ref, w_t_ref, offs_ref = x.clone(), w_t.clone(), offs.clone()

    # Quantize activations (M, K) -> packed (M, K//2), scales (M, K//16)
    x_packed, x_scales = _quantize_for_test(x)

    # Quantize weights: (E, N, K) with K on last dim for block-wise quantization
    w = w_t.transpose(-2, -1).contiguous()  # (E, K, N) -> (E, N, K)
    w_packed, w_scales = _quantize_3d_for_test(w)
    # w_packed shape: (E, N, K//2), w_scales shape: (E, N, K//16)

    # BF16 reference
    ref_out = torch._grouped_mm(x_ref, w_t_ref, offs=offs_ref, out_dtype=torch.bfloat16)

    # Emulated NVFP4: B_data=(E, N, K//2), B_scale=(E, N, K//16)
    out = _emulated_nvfp4_scaled_grouped_mm_2d_3d(
        x_packed, x_scales, w_packed, w_scales, offs=offs
    )

    # FP4 has much lower precision than FP8 (4 bits vs 8 bits),
    # so SQNR threshold is lower than MXFP8's 27.0 dB.
    sqnr = compute_error(ref_out, out)
    min_sqnr = 16.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


@skip_if_rocm("ROCm not supported")
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 2048), (1024, 2048, 4096)])
@pytest.mark.parametrize("num_experts", (1, 8, 16))
def test_emulated_nvfp4_grouped_gemm_2d_2d(M, K, N, num_experts):
    # Simulate 2d-2d grouped gemm: grad_weight = grad_output_t @ input
    # grad_output_t: (N, M), input: (M, K) -> result: (E, N, K)
    grad_out_t = torch.randn(N, M, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    offs = generate_jagged_offs(num_experts, M, multiple_of=BLOCK_SIZE)
    grad_out_t_ref, x_ref, offs_ref = (
        grad_out_t.clone(),
        x.clone(),
        offs.clone(),
    )

    # BF16 reference: (N, M) @ (M, K) = (E, N, K)
    ref_out = torch._grouped_mm(
        grad_out_t_ref, x_ref, offs=offs_ref, out_dtype=torch.bfloat16
    )

    # Quantize: grad_out_t is (N, M), quantized along last dim (M)
    grad_out_t_packed, grad_out_t_scales = _quantize_for_test(grad_out_t)
    # B follows MXFP8 convention: provided as (N, K), transposed internally.
    # x is (M, K), so transpose to (K, M) to serve as B=(K, M) -> B^T=(M, K).
    x_t = x.t().contiguous()  # (K, M)
    x_t_packed, x_t_scales = _quantize_for_test(x_t)

    # Emulated NVFP4: A=(N, M), B=(K, M) -> internally B^T=(M, K)
    # Result: (N, M) @ (M, K) = (E, N, K)
    out = _emulated_nvfp4_scaled_grouped_mm_2d_2d(
        grad_out_t_packed,
        grad_out_t_scales,
        x_t_packed,
        x_t_scales,
        offs=offs,
    )

    sqnr = compute_error(ref_out, out)
    min_sqnr = 16.0
    assert sqnr >= min_sqnr, f"sqnr {sqnr} is too low, must be >= {min_sqnr}"


def test_nvfp4_dequant_roundtrip():
    """Test that quantize -> dequantize preserves values approximately."""
    from torchao.prototype.moe_training.nvfp4_grouped_mm import (
        _nvfp4_dequantize,
    )

    torch.manual_seed(42)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    scales, packed = nvfp4_quantize(x, block_size=BLOCK_SIZE)
    x_recon = _nvfp4_dequantize(packed, scales, output_dtype=torch.bfloat16)

    assert x_recon.shape == x.shape
    sqnr = compute_error(x, x_recon)
    # Roundtrip only quantizes one tensor (vs GEMM quantizing both),
    # so accuracy is higher. Profiled min=20.0 across 3,200 runs.
    min_sqnr = 19.0
    assert sqnr >= min_sqnr, f"Roundtrip sqnr {sqnr} is too low, must be >= {min_sqnr}"


def test_nvfp4_dequant_roundtrip_with_per_tensor_scale():
    """Test that two-level scaling (block + per-tensor) dequantizes correctly."""
    from torchao.prototype.moe_training.nvfp4_grouped_mm import (
        _nvfp4_dequantize,
    )
    from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

    torch.manual_seed(42)
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    amax = x.abs().max()
    per_tensor_scale = per_tensor_amax_to_scale(amax)
    scales, packed = nvfp4_quantize(
        x, block_size=BLOCK_SIZE, per_tensor_scale=per_tensor_scale
    )
    x_recon = _nvfp4_dequantize(
        packed, scales, per_tensor_scale=per_tensor_scale, output_dtype=torch.bfloat16
    )

    assert x_recon.shape == x.shape
    sqnr = compute_error(x, x_recon)
    # Roundtrip only quantizes one tensor (vs GEMM quantizing both),
    # so accuracy is higher. Profiled min=20.0 across 3,200 runs.
    min_sqnr = 19.0
    assert sqnr >= min_sqnr, (
        f"Roundtrip sqnr with per_tensor_scale {sqnr} is too low, must be >= {min_sqnr}"
    )

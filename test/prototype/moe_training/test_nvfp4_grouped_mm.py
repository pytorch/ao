# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn import functional as F
from torch.utils._triton import has_triton

from torchao.utils import (
    is_MI300,
    is_MI350,
    is_sm_at_least_90,
    is_sm_at_least_100,
    torch_version_at_least,
)

if not (
    torch.cuda.is_available() and (is_sm_at_least_90() or is_MI300() or is_MI350())
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
from torchao.prototype.moe_training.nvfp4_training.hadamard_cutedsl_utils import (
    cutedsl_nvfp4_kernels_available,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference
from torchao.testing.utils import skip_if_rocm

if has_triton() and is_sm_at_least_100() and torch_version_at_least("2.10.0"):
    from torchao.prototype.moe_training.nvfp4_training import nvfp4_grouped_mm
    from torchao.prototype.moe_training.nvfp4_training.nvfp4_grouped_mm import (
        _resolve_backends,
        _to_nvfp4_rht_rs_then_scaled_grouped_mm,
    )

BLOCK_SIZE = 16

_KERNEL_PREFERENCES = [
    pytest.param(KernelPreference.AUTO, id="auto"),
    pytest.param(KernelPreference.TRITON, id="triton"),
    pytest.param(
        KernelPreference.CUTEDSL,
        marks=pytest.mark.skipif(
            not cutedsl_nvfp4_kernels_available(),
            reason="requires the CuteDSL runtime",
        ),
        id="cutedsl",
    ),
]


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


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
# Keep the ungrouped dim above 128 here. The columnwise scale buffer is
# (ungrouped // 128, tokens // 64, 32, 16), so at or below one 128-row tile the
# per-group and whole-extent block layouts are the same permutation and wgrad
# layout bugs are invisible at any expert count -- 8 experts at K=N=128 scores
# the same 15.06 dB with or without such a bug. Expert count is not the axis
# that gives this coverage; the ungrouped dim is.
@pytest.mark.parametrize("M,K,N", [(1024, 1024, 1024)])
@pytest.mark.parametrize("num_experts", (1, 8))
@pytest.mark.parametrize("kernel_preference", _KERNEL_PREFERENCES)
def test_nvfp4_grouped_gemm_fwd_bwd(M, K, N, num_experts, kernel_preference):
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    offs = generate_jagged_offs(num_experts, M, multiple_of=128, dtype=torch.int32)
    sign_vector = tuple(1 if i % 2 == 0 else -1 for i in range(16))
    sr_seed = torch.tensor([1234], dtype=torch.int64, device="cuda")

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    out_ref = torch._grouped_mm(
        x_ref,
        weight_ref.transpose(-2, -1),
        offs=offs.clone(),
        out_dtype=torch.bfloat16,
    )
    out = _to_nvfp4_rht_rs_then_scaled_grouped_mm(
        x,
        weight,
        sign_vector,
        sr_seed,
        offs=offs,
        pad_token_groups_for_grouped_mm=False,
        kernel_preference=kernel_preference,
    )

    assert out.shape == out_ref.shape == (M, N)
    output_sqnr = compute_error(out_ref, out)
    assert output_sqnr >= 15.0, f"Output SQNR {output_sqnr} is below 15.0"

    labels = torch.ones_like(out_ref)
    F.mse_loss(out_ref, labels).backward()
    F.mse_loss(out, labels).backward()

    assert x.grad.shape == x_ref.grad.shape == (M, K)
    input_grad_sqnr = compute_error(x_ref.grad, x.grad)
    assert input_grad_sqnr >= 14.0, f"Input grad SQNR {input_grad_sqnr} is below 14.0"

    assert weight.grad.shape == weight_ref.grad.shape == (num_experts, N, K)
    # One bound for any expert count. The multi-expert case had been relaxed to
    # 5.0, which hid a columnwise scale-layout bug in the wgrad GEMM that put it
    # at ~6.7 dB; that is fixed. 12.0 keeps this a regression test for it while
    # clearing the tightest legitimate shape (8 experts x 128 rows measures
    # 13.4 dB -- fewer tokens per expert means less averaging of quantization
    # noise). Single-expert shapes run ~15.5 dB.
    min_weight_grad_sqnr = 12.0
    weight_grad_sqnr = compute_error(weight_ref.grad, weight.grad)
    assert weight_grad_sqnr >= min_weight_grad_sqnr, (
        f"Weight grad SQNR {weight_grad_sqnr} is below {min_weight_grad_sqnr}"
    )


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
def test_resolve_backends_falls_back_without_cutedsl(monkeypatch):
    """AUTO degrades to Triton where CuteDSL cannot run; CUTEDSL says so instead."""
    monkeypatch.setattr(
        nvfp4_grouped_mm, "cutedsl_nvfp4_kernels_available", lambda: False
    )

    assert _resolve_backends(KernelPreference.AUTO, 8) == (False, False)
    assert _resolve_backends(KernelPreference.TRITON, 8) == (False, False)
    with pytest.raises(RuntimeError, match="CUTEDSL requires"):
        _resolve_backends(KernelPreference.CUTEDSL, 8)
    with pytest.raises(ValueError, match="AUTO, TRITON, or CUTEDSL"):
        _resolve_backends(KernelPreference.TORCH, 8)


@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
@pytest.mark.skipif(
    not cutedsl_nvfp4_kernels_available(), reason="requires the CuteDSL runtime"
)
def test_resolve_backends_is_per_op():
    """The expert cap belongs to the RHT ops alone: the weight quantize now accepts
    every shape the grouped GEMM does, so the RHT path falling back must not drag it."""
    assert _resolve_backends(KernelPreference.AUTO, 8) == (True, True)
    # 65 experts exceeds the CuteDSL group cap; the weight quantize is unaffected.
    assert _resolve_backends(KernelPreference.AUTO, 65) == (False, True)

    with pytest.raises(ValueError, match="at most 64 experts"):
        _resolve_backends(KernelPreference.CUTEDSL, 65)


@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(not has_triton(), reason="unsupported without triton")
@pytest.mark.skipif(not is_sm_at_least_100(), reason="Requires SM100+")
@pytest.mark.skipif(
    not torch_version_at_least("2.10.0"), reason="requires PyTorch 2.10+"
)
def test_nvfp4_grouped_gemm_unaligned_padding():
    # K=N=256, not 128: this is the only coverage of the padded/unaligned group
    # path, and that path depends on the padding restoring the 128-aligned group
    # boundaries the columnwise scale store requires. At 128 the scale layout is
    # a single row-tile and degenerate (see the note above the fwd/bwd shapes),
    # so the test cannot see a layout error in the path it exists to guard.
    M, K, N, num_experts = 256, 256, 256, 2
    torch.manual_seed(42)
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    weight = torch.randn(
        num_experts,
        N,
        K,
        dtype=torch.bfloat16,
        device="cuda",
        requires_grad=True,
    )
    offs = torch.tensor([64, M], dtype=torch.int32, device="cuda")
    sign_vector = tuple(1 if i % 2 == 0 else -1 for i in range(16))
    sr_seed = torch.tensor([1234], dtype=torch.int64, device="cuda")

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    out_ref = torch._grouped_mm(
        x_ref,
        weight_ref.transpose(-2, -1),
        offs=offs,
        out_dtype=torch.bfloat16,
    )
    out = _to_nvfp4_rht_rs_then_scaled_grouped_mm(
        x,
        weight,
        sign_vector,
        sr_seed,
        offs=offs,
        pad_token_groups_for_grouped_mm=True,
    )
    labels = torch.ones_like(out)
    F.mse_loss(out_ref, labels).backward()
    F.mse_loss(out, labels).backward()

    assert out.shape == out_ref.shape == (M, N)
    assert compute_error(out_ref, out) >= 15.0
    assert x.grad.shape == x_ref.grad.shape == (M, K)
    assert compute_error(x_ref.grad, x.grad) >= 14.0
    assert weight.grad.shape == weight_ref.grad.shape == (num_experts, N, K)
    assert compute_error(weight_ref.grad, weight.grad) >= 12.0


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

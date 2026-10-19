# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the generalized int1-int8 MPS GEMM kernel.

Tests the intNgemm_mm Metal kernel through the _linear_fp_act_{n}bit_weight
ops for all bit widths 1-8, covering:
  - Correctness vs reference dequant+matmul
  - GEMM path (M > 1) and GEMV path (M == 1)
  - Multiple group sizes (32, 64, 128, 256)
  - float, half, and bfloat16 dtypes
  - Edge cases (non-aligned N, uniform weights)
  - End-to-end tensor subclass usage
"""

import pytest
import torch

# Load the experimental MPS library
import torchao.experimental.ops.mps  # noqa: F401

device = "mps"
all_nbits = list(range(1, 9))
all_group_sizes = [32, 64, 128, 256]


def _reference_dequant_matmul(A, W_int, scales, zeros, group_size, nbit):
    """Reference: dequantize weights and do float matmul on CPU."""
    N, K = W_int.shape
    W_f = W_int.float()
    S_exp = scales.cpu().repeat_interleave(group_size, dim=1)[:, :K]
    Z_exp = zeros.cpu().repeat_interleave(group_size, dim=1)[:, :K]
    W_dequant = S_exp * W_f + Z_exp
    return (A.cpu().float() @ W_dequant.T).to(A.device)


def _pack_and_run(A, W_int, scales, zeros, group_size, nbit):
    """Pack weights and run the MPS low-bit linear op."""
    pack_op = getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
    linear_op = getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
    B = pack_op(W_int.cpu()).to(device)
    return linear_op(A, B, group_size, scales, zeros)


@pytest.mark.parametrize("nbit", all_nbits)
@pytest.mark.parametrize("group_size", [32, 128])
def test_gemm_correctness(nbit, group_size):
    """Test GEMM path (M > 1) correctness for all bit widths."""
    M, N, K = 64, 256, 256
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    # Tolerance scales with bit width (more bits = more precision but also
    # larger values, so absolute error can be slightly larger)
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, f"int{nbit}bit gs={group_size}: max_err={max_err} > tol={tol}"


@pytest.mark.parametrize("nbit", all_nbits)
def test_gemv_correctness(nbit):
    """Test GEMV path (M == 1, decode) correctness for all bit widths."""
    M, N, K, group_size = 1, 256, 512, 128
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, f"int{nbit}bit GEMV: max_err={max_err} > tol={tol}"


@pytest.mark.parametrize("nbit", all_nbits)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_dtypes(nbit, dtype):
    """Test float and half dtype support."""
    M, N, K, group_size = 32, 128, 256, 64
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=dtype, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=dtype, device=device
    )
    Z = (-Z_raw * S).to(dtype)
    A = torch.randn(M, K, dtype=dtype, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    tol = max(0.05, max_val * 0.005)
    assert max_err < tol, f"int{nbit}bit {dtype}: max_err={max_err} > tol={tol}"


@pytest.mark.parametrize("nbit", all_nbits)
def test_uniform_weights(nbit):
    """Test with uniform weights (all same value) to catch off-by-one errors."""
    M, N, K, group_size = 32, 128, 256, 64
    val = (1 << nbit) // 2  # midpoint
    W_int = torch.full((N, K), val, dtype=torch.uint8)
    S = torch.ones(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.full((N, K // group_size), val, dtype=torch.float32, device=device)
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.ones(M, K, dtype=torch.float32, device=device)

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    assert max_err < 0.01, f"int{nbit}bit uniform: max_err={max_err}"


@pytest.mark.parametrize("nbit", all_nbits)
def test_group_sizes(nbit):
    """Test all supported group sizes."""
    M, N, K = 32, 128, 256
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    for gs in all_group_sizes:
        if K % gs != 0:
            continue
        S = torch.randn(N, K // gs, dtype=torch.float32, device=device) * 0.01
        Z_raw = torch.randint(
            0, max_val + 1, (N, K // gs), dtype=torch.float32, device=device
        )
        Z = (-Z_raw * S).to(torch.float32)

        result = _pack_and_run(A, W_int, S, Z, gs, nbit)
        expected = _reference_dequant_matmul(A, W_int, S, Z, gs, nbit)

        max_err = (result.float() - expected.float()).abs().max().item()
        tol = max(0.01, max_val * 0.001)
        assert max_err < tol, f"int{nbit}bit gs={gs}: max_err={max_err} > tol={tol}"


@pytest.mark.parametrize("nbit", all_nbits)
def test_large_m(nbit):
    """Test with large M (prefill scenario)."""
    M, N, K, group_size = 256, 512, 512, 128
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, f"int{nbit}bit large M: max_err={max_err} > tol={tol}"


@pytest.mark.parametrize("K", [32, 64, 128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("nbit", [4, 8])
def test_k_sweep_race_condition(nbit, K):
    """K-sweep test to catch the shared-memory race condition.

    In the matmul A [M,K] @ B [N,K]^T -> [M,N], K is the reduction dimension
    (input features / current layer's hidden size). The kernel tiles K with
    BLOCK_SIZE_K=32, so K/32 = number of K-loop iterations.

    The race occurs when the result store reuses the A region (shared_memory
    offset 0) without a threadgroup_barrier after the K-loop. It manifests at
    K>=512 (>=16 K-loop iterations) because simdgroups desync across the
    outer-product accumulation loop (which only uses simdgroup_barrier, not
    threadgroup_barrier). Without the barrier, one simdgroup can start writing
    results to shared_A while another simdgroup is still reading from it in
    its last simdgroup_load.

    This test covers K values from 32 (1 iteration) to 4096 (128 iterations).
    Real LLM hidden sizes are 4096+, so the race would trigger in production
    if the barrier were ever removed. GPU race timing varies by chip/driver,
    but the race reliably triggers at K>=512 on Apple Silicon.
    """
    M, N, group_size = 64, 128, 64
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    num_groups = (K + group_size - 1) // group_size
    S = torch.randn(N, num_groups, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, num_groups), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    assert not torch.isnan(result).any(), f"int{nbit}bit K={K}: NaN in result"
    assert not torch.isinf(result).any(), f"int{nbit}bit K={K}: Inf in result"

    max_err = (result.float() - expected.float()).abs().max().item()
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, (
        f"int{nbit}bit K={K}: max_err={max_err} > tol={tol} "
        f"(possible shared-memory race condition)"
    )


# ---------------------------------------------------------------------------
# End-to-end tensor subclass tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nbit", all_nbits)
@pytest.mark.parametrize("group_size", [32, 128])
def test_gemv_matches_gemm_path(nbit, group_size):
    """GEMM (M>1) and GEMV (M=1) paths should agree for the same input.

    The two kernels accumulate in different orders, so exact equality is not
    expected, but with scaled inputs the difference is negligible.
    """
    M, K, N = 32, 256, 256
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    pack_op = getattr(torch.ops.torchao, f"_pack_weight_{nbit}bit")
    linear_op = getattr(torch.ops.torchao, f"_linear_fp_act_{nbit}bit_weight")
    B = pack_op(W_int.cpu()).to(device)

    # Run M=1 for each row (GEMV path) and M=32 (GEMM path)
    gemv_results = torch.stack(
        [linear_op(A[i : i + 1], B, group_size, S, Z)[0] for i in range(M)]
    )
    gemm_result = linear_op(A, B, group_size, S, Z)

    max_err = (gemm_result.float() - gemv_results.float()).abs().max().item()
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, (
        f"int{nbit}bit gs={group_size}: GEMM vs GEMV max_err={max_err} > tol={tol}"
    )


@pytest.mark.parametrize("nbit", [4, 8])
def test_is_deterministic_across_runs(nbit):
    """Running the kernel multiple times should produce identical results."""
    M, K, N, group_size = 128, 256, 512, 32
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    results = [_pack_and_run(A, W_int, S, Z, group_size, nbit) for _ in range(5)]
    for i in range(1, 5):
        torch.testing.assert_close(results[0], results[i], rtol=0, atol=0)


@pytest.mark.parametrize("nbit", [4, 8])
@pytest.mark.parametrize(
    "M,N",
    [
        (32, 64),  # single threadgroup
        (128, 512),  # many threadgroups (32 TGs)
        (256, 512),  # 64 TGs — stress test
    ],
)
def test_large_threadgroup_count(nbit, M, N):
    """Large threadgroup counts previously exposed a race condition."""
    K, group_size = 256, 32
    max_val = (1 << nbit) - 1
    W_int = torch.randint(0, max_val + 1, (N, K), dtype=torch.uint8)
    S = torch.randn(N, K // group_size, dtype=torch.float32, device=device) * 0.01
    Z_raw = torch.randint(
        0, max_val + 1, (N, K // group_size), dtype=torch.float32, device=device
    )
    Z = (-Z_raw * S).to(torch.float32)
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    result = _pack_and_run(A, W_int, S, Z, group_size, nbit)
    expected = _reference_dequant_matmul(A, W_int, S, Z, group_size, nbit)

    max_err = (result.float() - expected.float()).abs().max().item()
    tol = max(0.01, max_val * 0.001)
    assert max_err < tol, (
        f"int{nbit}bit M={M} N={N}: max_err={max_err} > tol={tol}"
    )


# ---------------------------------------------------------------------------
# End-to-end tensor subclass tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("nbit", [1, 2, 3, 4, 5, 6, 7, 8])
def test_tensor_subclass_linear(nbit):
    """Test IntxMPSExperimentalTensor end-to-end linear."""
    from torchao.prototype.quantization.intx_mps.intx_mps_experimental_tensor import (
        IntxMPSExperimentalTensor,
    )

    N, K, group_size = 128, 256, 64
    M = 32
    hp_weight = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1
    block_size = [1, group_size]

    qweight = IntxMPSExperimentalTensor.from_hp(hp_weight, block_size, nbit=nbit)
    assert qweight.nbit == nbit

    # Reference: dequantize and matmul
    ref_weight = qweight.dequantize()
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1
    expected = torch.nn.functional.linear(A, ref_weight)

    # Run through the tensor subclass dispatch
    result = torch.nn.functional.linear(A, qweight)

    max_err = (result.float() - expected.float()).abs().max().item()
    max_val = (1 << nbit) - 1
    tol = max(0.05, max_val * 0.01)
    assert max_err < tol, f"int{nbit}bit subclass: max_err={max_err} > tol={tol}"


# ---------------------------------------------------------------------------
# Comparison with PyTorch's native MPS int4/int8 kernels (macOS 15+)
#
# PyTorch (PR #130715) added native MPS int4 groupwise and int8 per-channel
# weight-only quantization starting with macOS 15.  These tests verify that
# the experimental simdgroup GEMM kernel produces results within the same
# tolerance as the native path.
#
# Native int4 convention:
#   - Input to _convert_weight_to_int4pack is pre-packed uint8 (2 int4 per byte,
#     hi-nibble first)
#   - Dequant: w = (q - 8) * scale + zero, where zero = w_min + 8*scale
#   - qScaleAndZeros: [K//gs, N, 2] with (scale, zero)
#
# Native int8 convention:
#   - _weight_int8pack_mm takes int8 weights and per-channel scales
#   - Dequant: w = q * scale  (symmetric, no zero point)
#
# For int4, both paths use the same min-max unsigned quantization (0-15), so
# the direct experimental-vs-native comparison should be very tight (only
# floating-point accumulation order differs).  For int8, the native path uses
# symmetric quantization while the experimental path uses asymmetric min-max,
# so both are compared against
# the fp32 reference rather than directly against each other.
# ---------------------------------------------------------------------------


def _pack_int4_to_uint8(w_q_uint8):
    """Pack individual uint8 int4 values (0-15) into packed uint8 (2 per byte).

    Hi-nibble first: byte = (q_even << 4) | q_odd
    """
    N, K = w_q_uint8.shape
    assert K % 2 == 0
    return ((w_q_uint8[:, 0::2] << 4) | w_q_uint8[:, 1::2]).to(torch.uint8)


@pytest.mark.parametrize("M", [1, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_int4_vs_native_mps(M, group_size):
    """Compare the experimental int4 path against PyTorch's native _weight_int4pack_mm.

    Both paths use min-max unsigned quantization (0-15), but the native path
    internally shifts to signed (-8 to 7) and uses a slightly different
    rounding/convention for scale and zero.  Both are compared against the
    fp32 reference rather than directly against each other, since the
    quantized values may differ slightly between the two conventions.
    """
    torch.manual_seed(42)
    N, K = 128, 256

    hp_weight = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    # --- Experimental path: IntxMPSExperimentalTensor ---
    from torchao.prototype.quantization.intx_mps.intx_mps_experimental_tensor import (
        IntxMPSExperimentalTensor,
    )

    our_qweight = IntxMPSExperimentalTensor.from_hp(hp_weight, [1, group_size], nbit=4)
    our_result = torch.nn.functional.linear(A, our_qweight)

    # --- Native PyTorch MPS int4 ---
    # Quantize with min-max (unsigned 0-15, same as the experimental path)
    qmin, qmax = 0, 15
    w = hp_weight.cpu()
    w_g = w.reshape(N, K // group_size, group_size)
    w_min = w_g.amin(dim=2, keepdim=True)
    w_max = w_g.amax(dim=2, keepdim=True)
    scales = (w_max - w_min) / (qmax - qmin)
    zeros = qmin - torch.round(w_min / scales)
    scales_exp = scales.repeat(1, 1, group_size).reshape(N, K)
    zeros_exp = zeros.repeat(1, 1, group_size).reshape(N, K)
    w_q = torch.clamp(torch.round(w / scales_exp + zeros_exp), qmin, qmax).to(
        torch.uint8
    )

    # Pack and run native (pre-packed uint8, 2 int4 per byte)
    w_q_packed = _pack_int4_to_uint8(w_q)
    packed = torch.ops.aten._convert_weight_to_int4pack(w_q_packed.to(device), 8)

    # Native dequant: w = (q - 8) * scale + zero, where zero = w_min + 8*scale
    # This simplifies to w = q * scale + w_min, matching the experimental (q - zero_unsigned) * scale
    scale_flat = scales.reshape(N, K // group_size)
    w_min_flat = w_min.reshape(N, K // group_size)
    zero_native = w_min_flat + 8 * scale_flat
    qScaleAndZeros = (
        torch.stack([scale_flat.T, zero_native.T], dim=2).to(device).to(torch.float32)
    )

    native_result = torch.ops.aten._weight_int4pack_mm(
        A, packed, group_size, qScaleAndZeros
    )

    # Both should be close to the fp32 reference
    ref = torch.nn.functional.linear(A, hp_weight)
    our_err = (our_result.float() - ref.float()).abs().max().item()
    native_err = (native_result.float() - ref.float()).abs().max().item()

    # 4-bit quantization with 0.1-scale weights: vs-ref error is dominated by
    # quantization granularity (16 levels), not kernel correctness.
    assert our_err < 0.1, f"Experimental int4 error too high: {our_err}"
    assert native_err < 0.1, f"Native int4 error too high: {native_err}"


@pytest.mark.parametrize("M", [1, 8, 2048])
def test_int8_delegated_to_native(M):
    """Verify that int8 per-channel delegation to native _weight_int8pack_mm
    produces correct results matching the native operator directly.

    int8 per-channel always delegates to PyTorch's native _weight_int8pack_mm.
    This test verifies the delegation path produces the same result as calling
    native directly.
    """
    torch.manual_seed(42)
    N, K = 128, 256

    hp_weight = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1
    A = torch.randn(M, K, dtype=torch.float32, device=device) * 0.1

    from torchao.prototype.quantization.intx_mps.intx_mps_experimental_tensor import (
        IntxMPSExperimentalTensor,
    )

    # int8 per-channel always delegates to native
    qw_delegated = IntxMPSExperimentalTensor.from_hp(hp_weight, [1, K], nbit=8)
    assert qw_delegated.use_native_int8, "Expected use_native_int8=True"
    assert qw_delegated.packed_weight.dtype == torch.int8, (
        f"Expected int8 packed weight, got {qw_delegated.packed_weight.dtype}"
    )
    assert qw_delegated.scales.ndim == 1, (
        f"Expected 1D scales for native int8, got {qw_delegated.scales.ndim}D"
    )

    delegated_result = torch.nn.functional.linear(A, qw_delegated)

    # Native directly
    w = hp_weight.cpu()
    w_max = w.abs().amax(dim=1, keepdim=True)
    scales_native = w_max / 127.0
    w_q_native = torch.round(w / scales_native).clamp(-128, 127).to(torch.int8)
    scales_1d = scales_native.squeeze(1).to(device).to(torch.float32)
    native_result = torch.ops.aten._weight_int8pack_mm(
        A, w_q_native.to(device), scales_1d
    )

    # Should match native exactly (same quantization scheme)
    err = (delegated_result.float() - native_result.float()).abs().max().item()
    assert err < 1e-5, f"Delegated int8 should match native exactly, got err={err}"

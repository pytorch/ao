# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from packaging import version
from torchao.float8.config import e4m3_dtype
from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.kernels import (
    EPS,
    torch_blockwise_scale_act_quant_lhs,
    torch_blockwise_scale_act_quant_rhs,
    torch_blockwise_scale_weight_quant,
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_act_quant_transposed_lhs_kernel,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x1,
    triton_fp8_gemm_1x128_128x128,
)
from torchao.utils import is_MI300, is_MI350, is_ROCM, is_sm_at_least_90

BLOCKWISE_SIZE_MNK = [
    # (128, 128, 128),
    (2, 512, 128),
    (2, 5120, 1280),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.skipif(is_ROCM(), reason="Blockwise FP8 GEMM has numerical issues on ROCm")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [e4m3_dtype])
def test_triton_fp8_gemm_1x128_128x128(M, N, K, dtype):
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    C = A @ B.T
    A_q, A_s = triton_fp8_blockwise_act_quant_lhs(A, dtype=dtype)
    B_t_q, B_t_s = triton_fp8_blockwise_weight_quant_transposed_rhs(B, dtype=dtype)
    C_q = triton_fp8_gemm_1x128_128x128(
        A_q, B_t_q, A_s, B_t_s, out_dtype=torch.bfloat16
    )
    assert not C_q.isnan().any(), "C_q must not contain NaNs"

    sqnr = compute_error(C, C_q)
    min_sqnr = 28.0
    assert sqnr >= min_sqnr, f"SQNR {sqnr:.2f} must be >= {min_sqnr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.skipif(is_ROCM(), reason="Blockwise FP8 GEMM has numerical issues on ROCm")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [e4m3_dtype])
def test_triton_fp8_gemm_1x128_128x1(M, N, K, dtype):
    A = torch.randn(K, M, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C = A.T @ B
    A_t_q, A_t_s = triton_fp8_blockwise_act_quant_transposed_lhs(A, dtype=dtype)
    B_q, B_s = triton_fp8_blockwise_act_quant_rhs(B, dtype=dtype)
    C_q = triton_fp8_gemm_1x128_128x1(A_t_q, B_q, A_t_s, B_s, out_dtype=torch.bfloat16)

    assert not C_q.isnan().any(), "C_q must not contain NaNs"
    assert C.dtype == torch.bfloat16
    assert C_q.dtype == torch.bfloat16

    sqnr = compute_error(C, C_q)
    min_sqnr = 28.0
    assert sqnr >= min_sqnr, f"SQNR {sqnr:.2f} must be >= {min_sqnr}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_act_quant_lhs(block_size):
    device = "cuda"
    M, K = 4096, 1024
    x = torch.randn(M, K, device=device)

    x[0, :block_size] = 0.0

    triton_fp8, triton_scale = triton_fp8_blockwise_act_quant_lhs(
        x,
        block_size=block_size,
    )

    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_lhs(x, tile_size=block_size)

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_act_quant_rhs(block_size: int):
    device = "cuda"
    M, K = 4096, 1024
    x = torch.randn(M, K, device=device)

    x[:block_size, :block_size] = 0.0

    triton_fp8, triton_scale = triton_fp8_blockwise_act_quant_rhs(
        x,
        block_size=block_size,
    )

    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_rhs(x, block_size=block_size)

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("M,K", [(4096, 1024), (4096, 4 * 4096)])
def test_triton_quantize_fp8_act_quant_transposed_lhs(M, K, block_size: int):
    device = "cuda"
    x = torch.randn(M, K, device=device)

    x[0, :block_size] = 0.0

    triton_fp8, triton_scale = triton_fp8_blockwise_act_quant_transposed_lhs(
        x,
        block_size=block_size,
    )

    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_lhs(
        x.t().contiguous(), tile_size=block_size
    )

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("num_groups", [2, 16, 128])
def test_triton_quantize_fp8_act_quant_transposed_lhs_wide_k(num_groups: int):
    """Regression test for the K axis being mapped onto gridY.

    The activation-gradient of a 70B-class lm_head is an (M=128, K=152064)
    tensor. With NUM_GROUPS=2 the K axis needs cdiv(152064, 2) = 76032 program
    tiles; CUDA caps gridY at 65535, so this shape failed to launch with
    "Triton Error [CUDA]: invalid argument" before K was moved to gridX.

    The wrapper does not forward NUM_GROUPS (it comes from @triton.autotune), so
    call the underlying jit function directly to pin the config.
    """
    device = "cuda"
    block_size = 128
    M, K = 128, 152064

    x = torch.randn(M, K, device=device)

    kern = triton_fp8_blockwise_act_quant_transposed_lhs_kernel
    fn = getattr(kern, "fn", kern)  # unwrap the autotuned kernel

    m_blocks = triton.cdiv(M, block_size)
    # Same output layout the wrapper produces: y is (K, M) and the scales are
    # (K, M // block_size) in column-major order.
    triton_fp8 = torch.empty(K, M, dtype=e4m3_dtype, device=device)
    triton_scale = torch.empty(
        K, m_blocks, dtype=torch.float32, device=device
    ).as_strided((K, m_blocks), (1, K))

    # K -> gridX, M -> gridY (the fixed mapping).
    grid = (triton.cdiv(K, num_groups), triton.cdiv(M, block_size))
    fn[grid](
        x,
        x.stride(0),
        x.stride(1),
        triton_fp8,
        triton_fp8.stride(0),
        triton_fp8.stride(1),
        triton_scale,
        triton_scale.stride(0),
        triton_scale.stride(1),
        M,
        K=K,
        BLOCK_SIZE=block_size,
        NUM_GROUPS=num_groups,
        EPS=EPS,
        FP8_MAX=torch.finfo(e4m3_dtype).max,
    )
    torch.cuda.synchronize()

    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_lhs(
        x.t().contiguous(), tile_size=block_size
    )

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"

    torch.testing.assert_close(
        triton_fp8.to(torch.float32),
        ref_fp8.to(torch.float32),
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp8.to(torch.float32) - ref_fp8.to(torch.float32)).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("M,K", [(4096, 1024), (4096, 4 * 4096)])
def test_triton_quantize_fp8_weight_quant_rhs(M, K, block_size: int):
    device = "cuda"
    x = torch.randn(M, K, device=device)

    x[:block_size, :block_size] = 0.0

    triton_fp8, triton_scale = triton_fp8_blockwise_weight_quant_rhs(
        x,
        block_size=block_size,
    )
    ref_fp8, ref_scale = torch_blockwise_scale_weight_quant(x, tile_size=block_size)

    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"

    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_weight_quant_transposed_rhs(block_size: int):
    device = "cuda"
    M = 512
    K = 2048
    x = torch.randn(M, K, device=device)

    x[:block_size, :block_size] = 0.0

    triton_fp8, triton_scale = triton_fp8_blockwise_weight_quant_transposed_rhs(
        x,
        block_size=block_size,
    )
    ref_fp8, ref_scale = torch_blockwise_scale_weight_quant(
        x.t().contiguous(), tile_size=block_size
    )

    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"

    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

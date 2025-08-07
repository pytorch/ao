# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from packaging import version
from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.kernels import (
    blockwise_fp8_gemm_1x128_128x1,
    blockwise_fp8_gemm_1x128_128x128,
    fp8_blockwise_act_quant_lhs,
    fp8_blockwise_act_quant_rhs,
    fp8_blockwise_act_quant_transposed_lhs,
    fp8_blockwise_weight_quant_rhs,
    fp8_blockwise_weight_quant_transposed_rhs,
    torch_blockwise_scale_act_quant_lhs,
    torch_blockwise_scale_act_quant_rhs,
    torch_blockwise_scale_weight_quant,
)
from torchao.testing.utils import skip_if_rocm
from torchao.utils import (
    is_sm_at_least_90,
    auto_detect_device,
)

_DEVICE = [auto_detect_device()]
print(11111111111111111111111111111, _DEVICE)

BLOCKWISE_SIZE_MNK = [
    (128, 128, 128),
    (2, 512, 128),
    (2, 5120, 1280),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]

@pytest.mark.parametrize("device", _DEVICE)
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_blockwise_fp8_gemm_1x128_128x128(device, M, N, K, dtype):
    # Simulate output = input @ weight.T
    A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    B = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant_lhs(A, dtype=dtype)
    B_t_q, B_t_s = fp8_blockwise_weight_quant_transposed_rhs(B, dtype=dtype)
    C_q = blockwise_fp8_gemm_1x128_128x128(A_q, 1.0 / A_s, B_t_q, 1.0 / B_t_s)
    assert not C_q.isnan().any(), "C_q must not contain NaNs"

    sqnr = compute_error(C, C_q)
    min_sqnr = 28.0
    assert sqnr >= min_sqnr, f"SQNR {sqnr:.2f} must be >= {min_sqnr}"

@pytest.mark.parametrize("device", _DEVICE)
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_blockwise_fp8_gemm_1x128_128x1(device, M, N, K, dtype):
    # Simulate grad_weight = grad_output_t @ input
    A = torch.randn(K, M, dtype=torch.bfloat16, device=device)
    B = torch.randn(K, N, dtype=torch.bfloat16, device=device)
    C = A.T @ B
    A_t_q, A_t_s = fp8_blockwise_act_quant_transposed_lhs(A, dtype=dtype)
    B_q, B_s = fp8_blockwise_act_quant_rhs(B, dtype=dtype)
    C_q = blockwise_fp8_gemm_1x128_128x1(A_t_q, 1.0 / A_t_s, B_q, 1.0 / B_s)

    assert not C_q.isnan().any(), "C_q must not contain NaNs"
    assert C.dtype == torch.bfloat16
    assert C_q.dtype == torch.bfloat16

    sqnr = compute_error(C, C_q)
    min_sqnr = 28.0
    assert sqnr >= min_sqnr, f"SQNR {sqnr:.2f} must be >= {min_sqnr}"

@pytest.mark.parametrize("device", _DEVICE)
@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(torch.cuda.is_available and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_act_quant_lhs(device, block_size):
    M, K = 4096, 1024
    x = torch.randn(M, K, device=device)

    # Set one scaling block to 0s, so if nan guards/EPS are not applied, the
    # quantized tensor will have NaNs due to division by 0
    x[0, :block_size] = 0.0

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = fp8_blockwise_act_quant_lhs(
        x,
        block_size=block_size,
    )

    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_lhs(x, tile_size=block_size)

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    # Compare scales
    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

@pytest.mark.parametrize("device", _DEVICE)
@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_act_quant_rhs(device, block_size: int):
    M, K = 4096, 1024
    x = torch.randn(M, K, device=device)

    # Set one block to 0s, so if nan guards/EPS are not applied, the
    # quantized tensor will have NaNs due to division by 0
    x[:block_size, :block_size] = 0.0

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = fp8_blockwise_act_quant_rhs(
        x,
        block_size=block_size,
    )

    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_rhs(x, block_size=block_size)

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    # Compare scales
    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

@pytest.mark.parametrize("device", _DEVICE)
@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("M,K", [(4096, 1024), (4096, 4 * 4096)])
def test_triton_quantize_fp8_act_quant_transposed_lhs(device, M, K, block_size: int):
    x = torch.randn(M, K, device=device)

    # Set one scaling block to 0s, so if nan guards/EPS are not applied, the
    # quantized tensor will have NaNs due to division by 0
    x[0, :block_size] = 0.0

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = fp8_blockwise_act_quant_transposed_lhs(
        x,
        block_size=block_size,
    )

    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_act_quant_lhs(
        x.t().contiguous(), tile_size=block_size
    )

    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    # Compare scales
    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

@pytest.mark.parametrize("device", _DEVICE)
@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("block_size", [128, 256])
@pytest.mark.parametrize("M,K", [(4096, 1024), (4096, 4 * 4096)])
def test_triton_quantize_fp8_weight_quant_rhs(device, M, K, block_size: int):
    x = torch.randn(M, K, device=device)

    # Set one scaling block to 0s, so if nan guards/EPS are not applied, the
    # quantized tensor will have NaNs due to division by 0
    x[:block_size, :block_size] = 0.0

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = fp8_blockwise_weight_quant_rhs(
        x,
        block_size=block_size,
    )
    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_weight_quant(x, tile_size=block_size)

    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    # Compare scales
    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

@pytest.mark.parametrize("device", _DEVICE)
@skip_if_rocm("ROCm not supported")
@pytest.mark.skipif(torch.cuda.is_available() and not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("block_size", [128, 256])
def test_triton_quantize_fp8_weight_quant_transposed_rhs(device, block_size: int):
    M = 512
    K = 2048
    x = torch.randn(M, K, device=device)

    # Set one scaling block to 0s, so if nan guards/EPS are not applied, the
    # quantized tensor will have NaNs due to division by 0
    x[:block_size, :block_size] = 0.0

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = fp8_blockwise_weight_quant_transposed_rhs(
        x,
        block_size=block_size,
    )
    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_weight_quant(
        x.t().contiguous(), tile_size=block_size
    )

    assert not ref_fp8.isnan().any(), "fp8 output must not contain NaNs"
    assert not triton_fp8.isnan().any(), "fp8 output must not contain NaNs"

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    torch.testing.assert_close(
        triton_fp32,
        ref_fp32,
        atol=0,
        rtol=0,
        msg=f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}",
    )

    # Compare scales
    torch.testing.assert_close(
        triton_scale,
        ref_scale,
        atol=0,
        rtol=0,
        msg=f"Scales differ: max diff = {(triton_scale - ref_scale).abs().max().item()}",
    )

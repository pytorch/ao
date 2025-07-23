# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from packaging import version
from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8.kernels import (
    torch_blockwise_scale_act_quant,
    torch_blockwise_scale_weight_quant,
    triton_quantize_fp8_block,
)
from torchao.testing.utils import skip_if_rocm

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8.kernels import (
    blockwise_fp8_gemm,
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_dequant,
    fp8_blockwise_weight_quant,
)
from torchao.utils import is_sm_at_least_89

BLOCKWISE_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("_, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
def test_blockwise_quant_dequant(_, N, K, dtype):
    x = torch.randn(N, K).cuda()
    qx, s = fp8_blockwise_weight_quant(x, dtype=dtype)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s)
    sqnr = compute_error(x, x_reconstructed)
    assert sqnr >= 25.0, f"SQNR {sqnr:.2f} must be >= 25.0"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
def test_blockwise_fp8_gemm(M, N, K, dtype):
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant(A, dtype=dtype)
    B_q, B_s = fp8_blockwise_weight_quant(B, dtype=dtype)
    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    sqnr = compute_error(C, C_q)
    assert sqnr >= 25.0, f"SQNR {sqnr:.2f} must be >= 25.0"


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("tile_size", [128, 256])
def test_triton_quantize_fp8_act_quant(tile_size: int):
    device = "cuda"
    M, K = 256, 256
    x = torch.randn(M, K, device=device)

    # Get the quantized tensor and scales using triton implementation
    # Use block_m=1 to match the narrow tiles (1 x tile_size) in the reference implementation
    triton_fp8, triton_scale = triton_quantize_fp8_block(
        x, block_m=1, block_k=tile_size
    )

    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_act_quant(x, tile_size=tile_size)

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    # Note: We use a relatively high tolerance because the implementations might have
    # slight differences in how they handle edge cases, rounding, etc.
    assert torch.allclose(triton_fp32, ref_fp32, rtol=1e-2, atol=1e-2), (
        f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}"
    )

    # Check that the scales are close
    # Note: The scales might be stored differently (reciprocal vs. direct), so we need to
    # be careful about how we compare them

    # In triton_quantize_fp8_block, scales are stored as reciprocals (1/scale)
    # In torch_blockwise_scale_act_quant, scales are stored directly
    # So we need to take the reciprocal of one of them for comparison

    # Reshape triton_scale to match ref_scale shape for comparison
    triton_scale_reshaped = triton_scale.reshape(M, -1)

    # Compare reciprocal of triton_scale with ref_scale
    assert torch.allclose(
        1.0 / triton_scale_reshaped, ref_scale, rtol=1e-2, atol=1e-2
    ), (
        f"Scales differ: max diff = {(1.0 / triton_scale_reshaped - ref_scale).abs().max().item()}"
    )


@skip_if_rocm("ROCm enablement in progress")
@pytest.mark.parametrize("tile_size", [128, 256])
def test_triton_quantize_fp8_weight_quant(tile_size: int):
    device = "cuda"
    # Make sure dimensions are multiples of tile_size for clean comparison
    M = tile_size * 2
    K = tile_size * 2
    x = torch.randn(M, K, device=device)

    # Get the quantized tensor and scales using triton implementation
    triton_fp8, triton_scale = triton_quantize_fp8_block(
        x, block_m=tile_size, block_k=tile_size
    )

    # Get the quantized tensor and scales using reference implementation
    ref_fp8, ref_scale = torch_blockwise_scale_weight_quant(x, tile_size=tile_size)

    # Convert both to float32 for comparison
    triton_fp32 = triton_fp8.to(torch.float32)
    ref_fp32 = ref_fp8.to(torch.float32)

    # Check that the quantized tensors are close
    assert torch.allclose(triton_fp32, ref_fp32, rtol=1e-2, atol=1e-2), (
        f"Quantized tensors differ: max diff = {(triton_fp32 - ref_fp32).abs().max().item()}"
    )

    # Check that the scales are close
    # In triton_quantize_fp8_block, scales are stored as reciprocals (1/scale)
    # In torch_blockwise_scale_weight_quant, scales are stored directly

    # Compare reciprocal of triton_scale with ref_scale
    assert torch.allclose(1.0 / triton_scale, ref_scale, rtol=1e-2, atol=1e-2), (
        f"Scales differ: max diff = {(1.0 / triton_scale - ref_scale).abs().max().item()}"
    )

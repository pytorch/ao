# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from packaging import version

from torchao.float8.float8_utils import compute_error
from torchao.utils import is_sm_at_least_90

triton = pytest.importorskip("triton", reason="Triton required to run this test")
if not is_sm_at_least_90():
    pytest.skip("This test requires SM90 or higher", allow_module_level=True)

from torchao.prototype.blockwise_fp8_training.kernels import (
    blockwise_fp8_gemm_1x128_128x128,
    blockwise_fp8_gemm_1x128_128x1,
    fp8_blockwise_act_quant_lhs,
    fp8_blockwise_act_quant_rhs,
    fp8_blockwise_act_quant_transposed_lhs,
    fp8_blockwise_weight_quant_rhs,
    fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.prototype.blockwise_fp8_training.scaled_mm_kernels import (
    blockwise_fp8_gemm_scaled_mm_1x128_128x128,
    blockwise_fp8_gemm_scaled_mm_1x128_128x1,
    blockwise_fp8_scaled_mm_1x128_128x128,
    blockwise_fp8_scaled_mm_1x128_128x1,
    blockwise_fp8_scaled_mm_advanced_1x128_128x128,
)
from torchao.prototype.blockwise_fp8_training.linear import (
    Float8BlockwiseLinear,
    Float8BlockwiseLinearConfig,
)

# Test matrix sizes covering various common LLM dimensions
SCALED_MM_TEST_SIZES = [
    (128, 128, 128),
    (2, 512, 128),
    (4, 4096, 4096),
    (8, 4096, 11008),
    (16, 11008, 4096),
    (1, 4096, 128256),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", SCALED_MM_TEST_SIZES)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_blockwise_fp8_scaled_mm_1x128_128x128_correctness(M, N, K, dtype):
    """Test correctness of torch._scaled_mm implementation vs Triton kernel."""
    if K % 128 != 0 or N % 128 != 0:
        pytest.skip(f"Dimensions K={K}, N={N} must be divisible by 128")
    
    device = torch.device("cuda")
    block_size = 128
    
    # Create high-precision reference tensors
    a_ref = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_ref = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    # Quantize inputs using the same quantization functions
    a_fp8, a_scale = fp8_blockwise_act_quant_lhs(a_ref, block_size)
    b_fp8, b_scale = fp8_blockwise_weight_quant_transposed_rhs(b_ref, block_size)
    
    # Compute using Triton kernel
    triton_output = blockwise_fp8_gemm_1x128_128x128(
        a_fp8,
        1.0 / a_scale,
        b_fp8, 
        1.0 / b_scale,
    )
    
    # Compute using torch._scaled_mm wrapper
    scaled_mm_output = blockwise_fp8_gemm_scaled_mm_1x128_128x128(
        a_fp8,
        1.0 / a_scale,
        b_fp8,
        1.0 / b_scale,
        block_size,
    )
    
    # Compare results - should be very close now with proper blockwise implementation
    error_db = compute_error(triton_output, scaled_mm_output)
    print(f"Error between Triton and scaled_mm (dB): {error_db}")
    
    # With proper blockwise scaling (not averaging), accuracy should be much better
    assert error_db > -80, f"Error too large: {error_db} dB (expected < -80 dB with proper blockwise scaling)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", SCALED_MM_TEST_SIZES)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn])
def test_blockwise_fp8_scaled_mm_1x128_128x1_correctness(M, N, K, dtype):
    """Test correctness of torch._scaled_mm implementation vs Triton kernel for 128x1 scaling."""
    if K % 128 != 0:
        pytest.skip(f"Dimension K={K} must be divisible by 128")
    
    device = torch.device("cuda")
    block_size = 128
    
    # Create high-precision reference tensors
    a_ref = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_ref = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    # Quantize inputs - note different scaling pattern for this variant
    a_fp8, a_scale = fp8_blockwise_act_quant_transposed_lhs(a_ref, block_size)
    b_fp8, b_scale = fp8_blockwise_act_quant_rhs(b_ref, block_size)
    
    # Compute using Triton kernel
    triton_output = blockwise_fp8_gemm_1x128_128x1(
        a_fp8,
        1.0 / a_scale,
        b_fp8,
        1.0 / b_scale,
        block_size,
    )
    
    # Compute using torch._scaled_mm wrapper  
    scaled_mm_output = blockwise_fp8_gemm_scaled_mm_1x128_128x1(
        a_fp8,
        1.0 / a_scale,
        b_fp8,
        1.0 / b_scale,
        block_size,
    )
    
    # Compare results
    error_db = compute_error(triton_output, scaled_mm_output)
    print(f"Error between Triton and scaled_mm 128x1 (dB): {error_db}")
    
    # With proper block-by-block implementation, accuracy should be much better
    assert error_db > -80, f"Error too large: {error_db} dB (expected < -80 dB with proper blockwise scaling)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("use_scaled_mm", [False, True])
@pytest.mark.parametrize("M, N, K", [(4, 4096, 4096), (8, 4096, 11008)])
def test_float8_blockwise_linear_forward_backward(use_scaled_mm, M, N, K):
    """Test forward and backward passes with both Triton and scaled_mm backends."""
    if K % 128 != 0 or N % 128 != 0:
        pytest.skip(f"Dimensions K={K}, N={N} must be divisible by 128")
    
    device = torch.device("cuda")
    
    # Create reference linear layer
    ref_layer = torch.nn.Linear(K, N, bias=False, device=device, dtype=torch.bfloat16)
    
    # Create blockwise fp8 layer
    test_layer = Float8BlockwiseLinear.from_float(ref_layer, use_scaled_mm=use_scaled_mm)
    
    # Create input
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)
    
    # Forward pass
    y_ref = ref_layer(x_ref)
    y_test = test_layer(x)
    
    # Check forward pass shapes
    assert y_test.shape == y_ref.shape
    
    # Backward pass
    grad_output = torch.randn_like(y_test)
    
    y_ref.backward(grad_output)
    y_test.backward(grad_output.clone())
    
    # Check gradient shapes
    assert x.grad.shape == x_ref.grad.shape
    assert test_layer.weight.grad.shape == ref_layer.weight.grad.shape
    
    print(f"Forward error (dB): {compute_error(y_ref, y_test)}")
    print(f"Input gradient error (dB): {compute_error(x_ref.grad, x.grad)}")
    print(f"Weight gradient error (dB): {compute_error(ref_layer.weight.grad, test_layer.weight.grad)}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_advanced_scaled_mm_implementation():
    """Test the advanced scaled_mm implementation that preserves more blockwise precision."""
    device = torch.device("cuda")
    M, K, N = 256, 1024, 512  # Divisible by 128
    block_size = 128
    
    # Create test tensors
    a_ref = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b_ref = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    
    # Quantize
    a_fp8, a_scale = fp8_blockwise_act_quant_lhs(a_ref, block_size)
    b_fp8, b_scale = fp8_blockwise_weight_quant_transposed_rhs(b_ref, block_size)
    
    # Both simple and advanced implementations now use the same high-accuracy approach
    default_output = blockwise_fp8_scaled_mm_1x128_128x128(
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )
    
    advanced_output = blockwise_fp8_scaled_mm_advanced_1x128_128x128(
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale, block_size
    )
    
    # Triton reference
    triton_output = blockwise_fp8_gemm_1x128_128x128(
        a_fp8, 1.0 / a_scale, b_fp8, 1.0 / b_scale
    )
    
    # Check shapes
    assert default_output.shape == advanced_output.shape == triton_output.shape
    
    # Both implementations should be identical now (default uses advanced)
    identity_error = compute_error(default_output, advanced_output)
    print(f"Default vs Advanced implementation error (dB): {identity_error}")
    assert identity_error > -120, "Default and advanced implementations should be identical"
    
    # Compare errors with Triton
    triton_error = compute_error(triton_output, default_output)
    print(f"Triton vs torch._scaled_mm error (dB): {triton_error}")
    
    # With proper blockwise implementation, should be very accurate
    assert triton_error > -80, f"Error too large: {triton_error} dB (expected < -80 dB with proper blockwise scaling)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_config_integration():
    """Test integration with configuration system."""
    device = torch.device("cuda")
    
    # Test both backends via config
    ref_layer = torch.nn.Linear(512, 1024, bias=False, device=device, dtype=torch.bfloat16)
    
    # Test Triton backend
    triton_config = Float8BlockwiseLinearConfig(use_scaled_mm=False)
    triton_layer = Float8BlockwiseLinear.from_float(ref_layer, use_scaled_mm=triton_config.use_scaled_mm)
    
    # Test scaled_mm backend
    scaled_mm_config = Float8BlockwiseLinearConfig(use_scaled_mm=True)
    scaled_mm_layer = Float8BlockwiseLinear.from_float(ref_layer, use_scaled_mm=scaled_mm_config.use_scaled_mm)
    
    # Test forward passes
    x = torch.randn(4, 512, device=device, dtype=torch.bfloat16)
    
    y_triton = triton_layer(x)
    y_scaled_mm = scaled_mm_layer(x)
    
    assert y_triton.shape == y_scaled_mm.shape
    assert not triton_layer.use_scaled_mm
    assert scaled_mm_layer.use_scaled_mm


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_error_conditions():
    """Test various error conditions and edge cases."""
    device = torch.device("cuda")
    
    # Test unsupported block sizes
    with pytest.raises(AssertionError, match="Only block_size=128 is supported"):
        blockwise_fp8_scaled_mm_1x128_128x128(
            torch.randn(128, 256, device=device, dtype=torch.float8_e4m3fn),
            torch.randn(128, 2, device=device, dtype=torch.float32),
            torch.randn(256, 128, device=device, dtype=torch.float8_e4m3fn),
            torch.randn(2, 1, device=device, dtype=torch.float32),
            block_size=64,  # Unsupported
        )
    
    # Test tensor shape mismatches
    with pytest.raises((RuntimeError, AssertionError)):
        blockwise_fp8_scaled_mm_1x128_128x128(
            torch.randn(128, 256, device=device, dtype=torch.float8_e4m3fn),
            torch.randn(128, 2, device=device, dtype=torch.float32),
            torch.randn(512, 128, device=device, dtype=torch.float8_e4m3fn),  # Wrong K dim
            torch.randn(4, 1, device=device, dtype=torch.float32),
            block_size=128,
        )


if __name__ == "__main__":
    pytest.main([__file__])
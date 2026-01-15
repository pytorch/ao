# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for MXFP4 Quantization-Aware Training (QAT) support.
"""

import copy

import pytest
import torch
import torch.nn as nn

from torchao.quantization.utils import compute_error
from torchao.utils import torch_version_at_least

torch.manual_seed(2)

if not torch_version_at_least("2.8.0"):
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.prototype.qat.mxfp4 import (
    MXFP4FakeQuantizeConfig,
    MXFP4FakeQuantizedLinear,
)
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    """Teardown dynamo cache after each test."""
    yield
    torch._dynamo.reset()


@pytest.mark.parametrize(
    "dtype,shape",
    [
        (torch.bfloat16, (32, 64)),
        (torch.float32, (64, 128)),
        (torch.bfloat16, (128, 256)),
        (torch.bfloat16, (1, 32, 64)),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_reconstruction(dtype, shape):
    """Test that MXFP4 quantization and dequantization works correctly."""
    x = torch.randn(shape, dtype=dtype, device="cuda")

    x_mxfp4 = MXTensor.to_mx(
        x,
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=32,
        scaling_mode=ScaleCalculationMode.FLOOR,
    )
    x_reconstructed = x_mxfp4.dequantize(dtype)

    def assert_sqnr_gt_threshold(orig, new, threshold):
        sqnr = compute_error(orig, new)
        if torch.all(torch.isnan(sqnr)):
            assert torch.all(orig == 0) and torch.all(new == 0)
        else:
            assert sqnr >= threshold

    # MXFP4 has lower precision than NVFP4, so we use a lower threshold
    assert_sqnr_gt_threshold(x, x_reconstructed, 6.0)

    assert x.shape == x_reconstructed.shape, (
        f"Shape mismatch: {x.shape} vs {x_reconstructed.shape}"
    )
    assert x.dtype == x_reconstructed.dtype, (
        f"Dtype mismatch: {x.dtype} vs {x_reconstructed.dtype}"
    )


@pytest.mark.parametrize(
    "scaling_mode",
    [
        ScaleCalculationMode.FLOOR,
        ScaleCalculationMode.CEIL,
        ScaleCalculationMode.EVEN,
        ScaleCalculationMode.RCEIL,
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_scaling_modes(scaling_mode):
    """Test MXFP4 with different scaling modes."""
    x = torch.randn(64, 128, dtype=torch.bfloat16, device="cuda")

    x_mxfp4 = MXTensor.to_mx(
        x,
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=32,
        scaling_mode=scaling_mode,
    )
    x_reconstructed = x_mxfp4.dequantize(torch.bfloat16)

    sqnr = compute_error(x, x_reconstructed)
    # All scaling modes should achieve reasonable SQNR
    assert sqnr >= 5.0, f"SQNR {sqnr:.2f} too low for scaling_mode={scaling_mode}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_fake_quantize_config():
    """Test MXFP4FakeQuantizeConfig dataclass."""
    config = MXFP4FakeQuantizeConfig()
    assert config.block_size == 32
    assert config.scaling_mode == ScaleCalculationMode.FLOOR
    assert config.kernel_preference == KernelPreference.EMULATED
    assert config.is_swizzled_scales is False

    config2 = MXFP4FakeQuantizeConfig(
        block_size=32,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED,
        is_swizzled_scales=True,
    )
    assert config2.block_size == 32
    assert config2.scaling_mode == ScaleCalculationMode.RCEIL
    assert config2.is_swizzled_scales is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("input_shape", [(128, 256), (1, 128, 256)])
@pytest.mark.parametrize(
    "scaling_mode",
    [ScaleCalculationMode.FLOOR, ScaleCalculationMode.RCEIL],
)
def test_mxfp4_fake_quantized_linear_forward(bias, input_shape, scaling_mode):
    """Test MXFP4FakeQuantizedLinear forward pass."""
    K, N = 256, 128

    activation_config = MXFP4FakeQuantizeConfig(
        block_size=32,
        scaling_mode=scaling_mode,
    )
    weight_config = MXFP4FakeQuantizeConfig(
        block_size=32,
        scaling_mode=scaling_mode,
    )

    linear = nn.Linear(K, N, bias=bias, device="cuda", dtype=torch.bfloat16)
    mxfp4_linear = MXFP4FakeQuantizedLinear.from_linear(
        linear,
        activation_config=activation_config,
        weight_config=weight_config,
    )

    x = torch.randn(*input_shape, device="cuda", dtype=torch.bfloat16)
    y_ref = linear(x)
    y_mxfp4 = mxfp4_linear(x)

    assert y_mxfp4.shape == y_ref.shape, (
        f"Shape mismatch: {y_mxfp4.shape} vs {y_ref.shape}"
    )
    assert y_mxfp4.dtype == y_ref.dtype, (
        f"Dtype mismatch: {y_mxfp4.dtype} vs {y_ref.dtype}"
    )

    # Check SQNR - MXFP4 has lower precision so use lower threshold
    sqnr = compute_error(y_ref, y_mxfp4)
    assert sqnr >= 5.0, f"SQNR {sqnr:.2f} too low for forward pass"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("bias", [True, False])
def test_mxfp4_fake_quantized_linear_backward(bias):
    """Test MXFP4FakeQuantizedLinear backward pass."""
    M, K, N = 128, 256, 128

    activation_config = MXFP4FakeQuantizeConfig(block_size=32)
    weight_config = MXFP4FakeQuantizeConfig(block_size=32)

    linear = nn.Linear(K, N, bias=bias, device="cuda", dtype=torch.bfloat16)
    linear_ref = copy.deepcopy(linear)
    mxfp4_linear = MXFP4FakeQuantizedLinear.from_linear(
        linear,
        activation_config=activation_config,
        weight_config=weight_config,
    )

    x_ref = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x = x_ref.clone().detach().requires_grad_(True)
    grad_output = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # Forward and backward for reference
    y_ref = linear_ref(x_ref)
    y_ref.backward(grad_output)

    # Forward and backward for MXFP4
    y_mxfp4 = mxfp4_linear(x)
    y_mxfp4.backward(grad_output)

    # Check that gradients are computed
    assert x.grad is not None, "Input gradient not computed"
    assert mxfp4_linear.weight.grad is not None, "Weight gradient not computed"

    # Check gradient shapes
    assert x.grad.shape == x_ref.grad.shape
    assert mxfp4_linear.weight.grad.shape == linear_ref.weight.grad.shape

    # Check gradient SQNR (expect lower due to quantization)
    x_grad_sqnr = compute_error(x_ref.grad, x.grad)
    w_grad_sqnr = compute_error(linear_ref.weight.grad, mxfp4_linear.weight.grad)

    # Lower threshold for gradients due to quantization effects
    assert x_grad_sqnr >= 3.0, f"Input grad SQNR {x_grad_sqnr:.2f} too low"
    assert w_grad_sqnr >= 3.0, f"Weight grad SQNR {w_grad_sqnr:.2f} too low"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_fake_quantized_linear_to_linear():
    """Test converting MXFP4FakeQuantizedLinear back to nn.Linear."""
    K, N = 256, 128

    activation_config = MXFP4FakeQuantizeConfig(block_size=32)
    weight_config = MXFP4FakeQuantizeConfig(block_size=32)

    original_linear = nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16)
    mxfp4_linear = MXFP4FakeQuantizedLinear.from_linear(
        original_linear,
        activation_config=activation_config,
        weight_config=weight_config,
    )

    # Convert back to linear
    converted_linear = mxfp4_linear.to_linear()

    assert isinstance(converted_linear, nn.Linear)
    assert converted_linear.in_features == K
    assert converted_linear.out_features == N
    assert converted_linear.bias is not None

    # Weights should be the same
    torch.testing.assert_close(converted_linear.weight, mxfp4_linear.weight)
    torch.testing.assert_close(converted_linear.bias, mxfp4_linear.bias)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_vs_nvfp4_block_size():
    """Test that MXFP4 uses different block size than NVFP4."""
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    # MXFP4 uses block_size=32 (OCP standard)
    x_mxfp4 = MXTensor.to_mx(
        x,
        elem_dtype=torch.float4_e2m1fn_x2,
        block_size=32,
        scaling_mode=ScaleCalculationMode.FLOOR,
    )

    # Check scale shape (should be (M, K // 32))
    expected_scale_shape = (128, 256 // 32)  # (128, 8)
    assert x_mxfp4.scale.shape == expected_scale_shape, (
        f"Scale shape mismatch: {x_mxfp4.scale.shape} vs {expected_scale_shape}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_config_error_handling():
    """Test error handling for MXFP4 config."""
    with pytest.raises(ValueError, match="Must specify `weight_config`"):
        MXFP4FakeQuantizedLinear(
            256,
            128,
            bias=True,
            activation_config=MXFP4FakeQuantizeConfig(),
            weight_config=None,
        )

    with pytest.raises(ValueError, match="Weight only MXFP4 QAT not supported yet"):
        MXFP4FakeQuantizedLinear(
            256,
            128,
            bias=True,
            activation_config=None,
            weight_config=MXFP4FakeQuantizeConfig(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "shapes",
    [
        (128, 64, 256),
        (256, 128, 512),
        (128, 128, 128),
    ],
    ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}",
)
def test_mxfp4_matmul_sqnr(shapes):
    """Test MXFP4 matrix multiplication SQNR."""
    M, K, N = shapes

    activation_config = MXFP4FakeQuantizeConfig(block_size=32)
    weight_config = MXFP4FakeQuantizeConfig(block_size=32)

    linear = nn.Linear(K, N, bias=False, device="cuda", dtype=torch.bfloat16)
    mxfp4_linear = MXFP4FakeQuantizedLinear.from_linear(
        linear,
        activation_config=activation_config,
        weight_config=weight_config,
    )

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    y_ref = linear(x)
    y_mxfp4 = mxfp4_linear(x)

    sqnr = compute_error(y_ref, y_mxfp4)
    SQNR_THRESHOLD = 5.0  # Lower threshold for MXFP4

    assert sqnr >= SQNR_THRESHOLD, (
        f"SQNR {sqnr:.2f} < {SQNR_THRESHOLD} for shapes {shapes}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_mxfp4_training_simulation():
    """Simulate a simple training loop with MXFP4 QAT."""
    M, K, N = 128, 256, 128
    num_steps = 5

    activation_config = MXFP4FakeQuantizeConfig(block_size=32)
    weight_config = MXFP4FakeQuantizeConfig(block_size=32)

    model = nn.Sequential(
        nn.Linear(K, N, bias=True, device="cuda", dtype=torch.bfloat16),
    )

    # Save initial weight BEFORE creating mxfp4_model (they share the same weight tensor)
    initial_weight = model[0].weight.clone()

    # Convert to MXFP4 QAT
    mxfp4_model = nn.Sequential(
        MXFP4FakeQuantizedLinear.from_linear(
            model[0],
            activation_config=activation_config,
            weight_config=weight_config,
        ),
    )

    optimizer = torch.optim.SGD(mxfp4_model.parameters(), lr=0.01)

    # Training loop
    for _ in range(num_steps):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        target = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

        optimizer.zero_grad()
        output = mxfp4_model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    # Check that weights have been updated
    assert not torch.allclose(mxfp4_model[0].weight, initial_weight), (
        "Weights should be updated during training"
    )

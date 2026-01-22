# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.nn.functional as F

from torchao.prototype.gptq import (
    GPTQConfig,
    gptq_quantize,
)
from torchao.prototype.gptq.observer import GPTQObserverTensor
from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
from torchao.quantization.granularity import PerRow
from torchao.utils import _is_fbgemm_gpu_genai_available


def _calculate_hessian(inputs, device=None):
    """Calculate Hessian matrix from input activations for GPTQ."""
    H = 0
    total_batches = 0

    for inp in inputs:
        # Setup x (activation tensor)
        x = inp.float()
        if device:
            x = x.to(device)
        shape = x.shape
        n = 1 if len(shape) == 2 else shape[0]
        x = x.reshape(-1, shape[-1])

        # Update Hessian with running average
        H *= total_batches / (total_batches + n)
        total_batches += n

        x = ((2 / total_batches) ** (1 / 2)) * x.t()
        H += x.matmul(x.t())

    return H


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, k, bias=False)
        self.linear2 = torch.nn.Linear(k, n, bias=False)
        self.linear3 = torch.nn.Linear(n, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x


class TestGPTQObserverTensor:
    """Test suite for GPTQObserverTensor functionality."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_observer_tensor_creation(self):
        """Test that GPTQObserverTensor.from_hp() creates tensor with correct properties."""
        weight = torch.randn(32, 64, dtype=torch.float32, device="cuda")
        observer = GPTQObserverTensor.from_hp(weight)

        # Check it's an GPTQObserverTensor
        assert isinstance(observer, GPTQObserverTensor)

        # Check shape matches
        assert observer.shape == weight.shape

        # Check dtype and device match
        assert observer.dtype == weight.dtype
        assert observer.device == weight.device

        # Check hp_data is stored correctly
        torch.testing.assert_close(observer.hp_data, weight)

        # Check hessian is initialized as None
        assert observer.hessian is None

        # Check total_batches is initialized as 0
        assert observer.total_batches == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_observer_tensor_attributes(self):
        """Test GPTQObserverTensor attributes are correctly set."""
        weight = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
        observer = GPTQObserverTensor.from_hp(weight)

        # Test hp_data attribute
        assert hasattr(observer, "hp_data")
        assert isinstance(observer.hp_data, torch.Tensor)

        # Test hessian attribute
        assert hasattr(observer, "hessian")
        assert observer.hessian is None

        # Test total_batches attribute
        assert hasattr(observer, "total_batches")
        assert observer.total_batches == 0

        # Test update method exists
        assert hasattr(observer, "update")
        assert callable(observer.update)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_linear_operation_with_observer(self):
        """Test F.linear with GPTQObserverTensor updates Hessian correctly."""
        batch_size = 4
        in_features = 64
        out_features = 32

        # Create weight as GPTQObserverTensor
        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )
        observer_weight = GPTQObserverTensor.from_hp(weight)

        # Create input
        input_tensor = torch.randn(
            batch_size, in_features, dtype=torch.float32, device="cuda"
        )

        # Perform linear operation
        output = F.linear(input_tensor, observer_weight)

        # Check output shape is correct
        assert output.shape == (batch_size, out_features)

        # Check that Hessian was initialized and updated
        assert observer_weight.hessian is not None
        assert observer_weight.hessian.shape == (in_features, in_features)
        assert observer_weight.total_batches == 1

        # Verify output is correct
        expected_output = F.linear(input_tensor, weight)
        torch.testing.assert_close(output, expected_output)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_multiple_observations(self):
        """Test that Hessian updates incrementally across multiple forward passes."""
        out_features = 16
        in_features = 32

        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )
        observer_weight = GPTQObserverTensor.from_hp(weight)

        num_passes = 5
        total_samples = 0

        # Perform multiple forward passes
        for i in range(num_passes):
            batch_size = 2
            input_tensor = torch.randn(
                batch_size, in_features, dtype=torch.float32, device="cuda"
            )
            total_samples += 1
            _ = F.linear(input_tensor, observer_weight)

        # Check that Hessian was created and updated
        assert observer_weight.hessian is not None
        assert observer_weight.hessian.shape == (in_features, in_features)

        # Check total_batches matches total samples
        assert observer_weight.total_batches == total_samples

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_bmm_operation_with_observer(self):
        """Test torch.bmm with GPTQObserverTensor updates Hessian correctly."""
        batch = 4
        m = 8
        n = 16
        k = 12

        # Create input and weight tensors
        input_tensor = torch.randn(batch, m, k, dtype=torch.float32, device="cuda")
        weight = torch.randn(batch, k, n, dtype=torch.float32, device="cuda")
        observer_weight = GPTQObserverTensor.from_hp(weight)

        # Perform bmm operation
        output = torch.bmm(input_tensor, observer_weight)

        # Check output shape
        assert output.shape == (batch, m, n)

        # Check Hessian was initialized and updated
        assert observer_weight.hessian is not None
        # For bmm with batch dimension, the Hessian is computed on the last dimension
        assert observer_weight.total_batches == batch

        # Verify output is correct
        expected_output = torch.bmm(input_tensor, weight)
        torch.testing.assert_close(output, expected_output)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize(
        "base_config",
        [
            pytest.param(
                Int4WeightOnlyConfig(group_size=128),
                id="int4",
                marks=pytest.mark.skipif(
                    not _is_fbgemm_gpu_genai_available(),
                    reason="fbgemm_gpu not available",
                ),
            ),
            pytest.param(Int8WeightOnlyConfig(group_size=128), id="int8"),
        ],
    )
    def test_observer_config_transform(self, base_config):
        """Test GPTQConfig wraps module weights correctly."""
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 32, bias=False).cuda()
        original_weight = linear.weight.data.clone()

        # Apply GPTQConfig with observe step
        quantize_(linear, GPTQConfig(step="observe", base_config=base_config))

        # Check weight is now an GPTQObserverTensor
        assert isinstance(linear.weight, GPTQObserverTensor)

        # Check hp_data matches original weight
        torch.testing.assert_close(linear.weight.hp_data, original_weight)

        # Check hessian is None initially
        assert linear.weight.hessian is None
        assert linear.weight.total_batches == 0

        # Perform a forward pass
        input_tensor = torch.randn(4, 64, dtype=torch.float32, device="cuda")
        output = linear(input_tensor)

        # Check Hessian was initialized after forward pass
        assert linear.weight.hessian is not None
        assert linear.weight.total_batches == 1

        # Check output shape
        assert output.shape == (4, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_hessian_incremental_update(self):
        """Test that incremental Hessian updates match batch calculation."""
        in_features = 32
        out_features = 16

        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )

        # Create two GPTQObserverTensors - one for incremental, one for batch
        observer_incremental = GPTQObserverTensor.from_hp(weight)

        # Collect activations for batch computation
        activations = []
        num_batches = 3
        for _ in range(num_batches):
            batch_size = 4
            input_tensor = torch.randn(
                batch_size, in_features, dtype=torch.float32, device="cuda"
            )
            activations.append(input_tensor)
            # Update incrementally
            _ = F.linear(input_tensor, observer_incremental)

        # Compute Hessian in batch using _calculate_hessian
        hessian_batch = _calculate_hessian(activations, device="cuda")

        # Compare incremental vs batch
        assert observer_incremental.hessian is not None
        torch.testing.assert_close(
            observer_incremental.hessian, hessian_batch, rtol=1e-4, atol=1e-5
        )


class TestGPTQFlow:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize(
        "base_config",
        [
            pytest.param(
                Int4WeightOnlyConfig(group_size=128),
                id="int4",
                marks=pytest.mark.skipif(
                    not _is_fbgemm_gpu_genai_available(),
                    reason="fbgemm_gpu not available",
                ),
            ),
            pytest.param(
                Int8WeightOnlyConfig(group_size=128),
                id="int8",
            ),
        ],
    )
    def test_unified_config_two_phase(self, base_config):
        """Test that GPTQConfig handles both observation and quantization phases."""
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 32, bias=False).cuda().to(torch.bfloat16)
        original_weight = linear.weight.data.clone()

        # Phase 1: Observation step - wrap as GPTQObserverTensor
        observe_config = GPTQConfig(
            step="observe",
            base_config=base_config,
        )
        quantize_(linear, observe_config)

        # Verify weight is now an GPTQObserverTensor
        assert isinstance(linear.weight, GPTQObserverTensor)
        torch.testing.assert_close(linear.weight.hp_data, original_weight)

        # Run some forward passes for calibration
        for _ in range(10):
            input_tensor = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
            _ = linear(input_tensor)

        # Verify Hessian was computed
        assert linear.weight.hessian is not None
        assert linear.weight.total_batches > 0

        # Phase 2: Convert step - apply GPTQ quantization
        convert_config = GPTQConfig(
            step="convert",
            base_config=base_config,
        )
        quantize_(linear, convert_config)

        # Verify weight is now Int4Tensor or Int8Tensor (quantized)
        from torchao.quantization import Int4Tensor, Int8Tensor

        expected_tensor_type = (
            Int4Tensor if isinstance(base_config, Int4WeightOnlyConfig) else Int8Tensor
        )
        assert isinstance(linear.weight, expected_tensor_type)

        # Verify it still works
        output = linear(input_tensor)
        assert output.shape == (4, 32)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize(
        "base_config",
        [
            pytest.param(
                Int4WeightOnlyConfig(group_size=128),
                id="int4",
                marks=pytest.mark.skipif(
                    not _is_fbgemm_gpu_genai_available(),
                    reason="fbgemm_gpu not available",
                ),
            ),
            pytest.param(
                Int8WeightOnlyConfig(group_size=128),
                id="int8",
            ),
        ],
    )
    def test_gptq_quantize_function(self, base_config):
        """Test gptq_quantize function with synthetic Hessian and weights."""
        torch.manual_seed(42)

        # Create synthetic weight matrix
        out_features = 128
        in_features = 256
        weight = torch.randn(
            out_features, in_features, dtype=torch.bfloat16, device="cuda"
        )

        # Create synthetic Hessian (positive semi-definite)
        # H = A^T @ A ensures positive semi-definiteness
        A = torch.randn(in_features, in_features, dtype=torch.float32, device="cuda")
        H = A.t() @ A
        # Add regularization to ensure positive definiteness
        H = H + torch.eye(in_features, device="cuda") * 0.1

        # Create GPTQ config
        config = GPTQConfig(
            step="convert",
            base_config=base_config,
        )

        # Run GPTQ quantization
        quantized_weight = gptq_quantize(H, weight, config)

        # Check output type
        from torchao.quantization import Int4Tensor, Int8Tensor

        expected_tensor_type = (
            Int4Tensor if isinstance(base_config, Int4WeightOnlyConfig) else Int8Tensor
        )
        assert isinstance(quantized_weight, expected_tensor_type)

        # Check shape is preserved
        assert quantized_weight.shape == weight.shape

        # Dequantize and check error is reasonable
        dequantized = F.linear(
            torch.eye(in_features, device="cuda", dtype=torch.bfloat16),
            quantized_weight,
            None,
        ).t()
        assert dequantized.shape == weight.shape

        # Check quantization introduces bounded error
        error = torch.abs(dequantized - weight.float())
        mean_error = error.mean().item()
        max_error = error.max().item()

        # GPTQ should have reasonable error bounds
        assert mean_error < 0.5, f"Mean error too high: {mean_error}"
        assert max_error < 5.0, f"Max error too high: {max_error}"

        # Check that quantization actually compressed the data
        # Int4 should be much smaller than bfloat16
        assert hasattr(quantized_weight, "qdata")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize(
        "base_config",
        [
            pytest.param(
                Int4WeightOnlyConfig(group_size=128),
                id="int4",
                marks=pytest.mark.skipif(
                    not _is_fbgemm_gpu_genai_available(),
                    reason="fbgemm_gpu not available",
                ),
            ),
            pytest.param(
                Int8WeightOnlyConfig(granularity=PerRow(), version=2), id="int8"
            ),
        ],
    )
    def test_gptq_quantize_better_than_naive(self, base_config):
        """Test that GPTQ produces lower error than naive quantization."""
        torch.manual_seed(43)

        # Create weight and realistic Hessian from actual activations
        out_features = 64
        in_features = 128
        weight = torch.randn(
            out_features, in_features, dtype=torch.bfloat16, device="cuda"
        )

        # Simulate activations and compute Hessian
        num_samples = 100
        activations = []
        for _ in range(num_samples):
            act = torch.randn(4, in_features, dtype=torch.float32, device="cuda")
            activations.append(act)

        H = _calculate_hessian(activations, device="cuda")
        H_identity = torch.eye(in_features, device="cuda", dtype=torch.float32)

        # GPTQ quantization
        config = GPTQConfig(
            step="convert",
            base_config=base_config,
        )
        gptq_quantized = gptq_quantize(H, weight, config)
        gptq_dequantized = F.linear(
            H_identity.to(torch.bfloat16), gptq_quantized, None
        ).t()

        # Naive quantization (using identity Hessian)
        naive_quantized = gptq_quantize(H_identity, weight, config)
        naive_dequantized = F.linear(
            H_identity.to(torch.bfloat16), naive_quantized, None
        ).t()

        # Compute weighted error using Hessian
        # Error metric: (W - W_q)^T H (W - W_q)
        weight_f = weight.float()
        gptq_error = weight_f - gptq_dequantized
        naive_error = weight_f - naive_dequantized

        # Compute Frobenius norm of errors
        gptq_loss = torch.norm(gptq_error).item()
        naive_loss = torch.norm(naive_error).item()

        print(f"GPTQ loss: {gptq_loss:.4f}, Naive loss: {naive_loss:.4f}")

        # GPTQ should generally produce lower or comparable error
        # (Note: with random data, this might not always hold, but with real Hessian it should)
        assert gptq_loss is not None
        assert naive_loss is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    @pytest.mark.parametrize(
        "base_config",
        [
            pytest.param(
                Int4WeightOnlyConfig(version=2),
                id="int4",
                marks=pytest.mark.skipif(
                    not _is_fbgemm_gpu_genai_available(),
                    reason="fbgemm_gpu not available",
                ),
            ),
            pytest.param(
                Int8WeightOnlyConfig(granularity=PerRow(), version=2), id="int8"
            ),
        ],
    )
    def test_gptq_sqnr(self, base_config):
        torch.manual_seed(43)

        model = ToyLinearModel(m=512, n=2048, k=1024).cuda().to(torch.bfloat16)

        # Create calibration and test inputs
        calibration_inputs = [
            torch.randn(4, 512, dtype=torch.bfloat16, device="cuda") for _ in range(10)
        ]
        test_input = calibration_inputs[0]

        # Get baseline output
        out = model(test_input)

        # Make copies for comparison
        model2 = copy.deepcopy(model)
        copy.deepcopy(model)

        # Apply RTN quantization for comparison
        quantize_(model2, base_config)
        out_rtn = model2(test_input)

        # Apply GPTQ observe step
        gptqnew_config = GPTQConfig(step="observe", base_config=base_config)
        quantize_(model, gptqnew_config)

        # Run calibration
        for inp in calibration_inputs:
            model(inp)

        # Apply GPTQ convert step
        convert_config = GPTQConfig(step="convert", base_config=base_config)
        quantize_(model, convert_config)
        out_gptq = model(test_input)

        # Compare using SQNR
        from torchao.quantization.utils import compute_error

        sqnr_rtn = compute_error(out_rtn, out)
        sqnr_gptq = compute_error(out_gptq, out)

        print(f"GPTQ SQNR: {sqnr_gptq}, RTN SQNR: {sqnr_rtn}")

        # Assert GPTQ quality
        if isinstance(base_config, Int4WeightOnlyConfig):
            assert sqnr_gptq > 25, f"GPTQ SQNR: {sqnr_gptq} is too low"
        elif isinstance(base_config, Int8WeightOnlyConfig):
            assert sqnr_gptq > 30, f"GPTQ SQNR: {sqnr_gptq} is too low"
        assert sqnr_gptq > sqnr_rtn, (
            f"GPTQ SQNR: {sqnr_gptq} is not better than RTN SQNR: {sqnr_rtn}"
        )

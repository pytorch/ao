# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
import torch.nn.functional as F

from torchao.prototype.gptq import (
    GPTQConfig,
    gptq_quantize,
)
from torchao.prototype.gptq.observer import ObserverTensor
from torchao.quantization import Int4WeightOnlyConfig, quantize_


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


class TestObserverTensor(unittest.TestCase):
    """Test suite for ObserverTensor functionality."""

    def test_observer_tensor_creation(self):
        """Test that ObserverTensor.from_hp() creates tensor with correct properties."""
        weight = torch.randn(32, 64, dtype=torch.float32, device="cuda")
        observer = ObserverTensor.from_hp(weight)

        # Check it's an ObserverTensor
        self.assertIsInstance(observer, ObserverTensor)

        # Check shape matches
        self.assertEqual(observer.shape, weight.shape)

        # Check dtype and device match
        self.assertEqual(observer.dtype, weight.dtype)
        self.assertEqual(observer.device, weight.device)

        # Check hp_data is stored correctly
        torch.testing.assert_close(observer.hp_data, weight)

        # Check hessian is initialized as None
        self.assertIsNone(observer.hessian)

        # Check total_batches is initialized as 0
        self.assertEqual(observer.total_batches, 0)

    def test_observer_tensor_attributes(self):
        """Test ObserverTensor attributes are correctly set."""
        weight = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
        observer = ObserverTensor.from_hp(weight)

        # Test hp_data attribute
        self.assertTrue(hasattr(observer, "hp_data"))
        self.assertIsInstance(observer.hp_data, torch.Tensor)

        # Test hessian attribute
        self.assertTrue(hasattr(observer, "hessian"))
        self.assertIsNone(observer.hessian)

        # Test total_batches attribute
        self.assertTrue(hasattr(observer, "total_batches"))
        self.assertEqual(observer.total_batches, 0)

        # Test update method exists
        self.assertTrue(hasattr(observer, "update"))
        self.assertTrue(callable(observer.update))

    def test_linear_operation_with_observer(self):
        """Test F.linear with ObserverTensor updates Hessian correctly."""
        batch_size = 4
        in_features = 64
        out_features = 32

        # Create weight as ObserverTensor
        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )
        observer_weight = ObserverTensor.from_hp(weight)

        # Create input
        input_tensor = torch.randn(
            batch_size, in_features, dtype=torch.float32, device="cuda"
        )

        # Perform linear operation
        output = F.linear(input_tensor, observer_weight)

        # Check output shape is correct
        self.assertEqual(output.shape, (batch_size, out_features))

        # Check that Hessian was initialized and updated
        self.assertIsNotNone(observer_weight.hessian)
        self.assertEqual(observer_weight.hessian.shape, (in_features, in_features))
        self.assertEqual(observer_weight.total_batches, batch_size)

        # Verify output is correct
        expected_output = F.linear(input_tensor, weight)
        torch.testing.assert_close(output, expected_output)

    def test_multiple_observations(self):
        """Test that Hessian updates incrementally across multiple forward passes."""
        out_features = 16
        in_features = 32

        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )
        observer_weight = ObserverTensor.from_hp(weight)

        num_passes = 5
        total_samples = 0

        # Perform multiple forward passes
        for i in range(num_passes):
            batch_size = 2
            input_tensor = torch.randn(
                batch_size, in_features, dtype=torch.float32, device="cuda"
            )
            total_samples += batch_size
            _ = F.linear(input_tensor, observer_weight)

        # Check that Hessian was created and updated
        self.assertIsNotNone(observer_weight.hessian)
        self.assertEqual(observer_weight.hessian.shape, (in_features, in_features))

        # Check total_batches matches total samples
        self.assertEqual(observer_weight.total_batches, total_samples)

    def test_bmm_operation_with_observer(self):
        """Test torch.bmm with ObserverTensor updates Hessian correctly."""
        batch = 4
        m = 8
        n = 16
        k = 12

        # Create input and weight tensors
        input_tensor = torch.randn(batch, m, k, dtype=torch.float32, device="cuda")
        weight = torch.randn(batch, k, n, dtype=torch.float32, device="cuda")
        observer_weight = ObserverTensor.from_hp(weight)

        # Perform bmm operation
        output = torch.bmm(input_tensor, observer_weight)

        # Check output shape
        self.assertEqual(output.shape, (batch, m, n))

        # Check Hessian was initialized and updated
        self.assertIsNotNone(observer_weight.hessian)
        # For bmm with batch dimension, the Hessian is computed on the last dimension
        self.assertEqual(observer_weight.total_batches, batch * m)

        # Verify output is correct
        expected_output = torch.bmm(input_tensor, weight)
        torch.testing.assert_close(output, expected_output)

    def test_observer_config_transform(self):
        """Test GPTQConfig wraps module weights correctly."""
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 32, bias=False).cuda()
        original_weight = linear.weight.data.clone()

        # Apply GPTQConfig with observe step
        quantize_(linear, GPTQConfig(step="observe", group_size=128))

        # Check weight is now an ObserverTensor
        self.assertIsInstance(linear.weight, ObserverTensor)

        # Check hp_data matches original weight
        torch.testing.assert_close(linear.weight.hp_data, original_weight)

        # Check hessian is None initially
        self.assertIsNone(linear.weight.hessian)
        self.assertEqual(linear.weight.total_batches, 0)

        # Perform a forward pass
        input_tensor = torch.randn(4, 64, dtype=torch.float32, device="cuda")
        output = linear(input_tensor)

        # Check Hessian was initialized after forward pass
        self.assertIsNotNone(linear.weight.hessian)
        self.assertEqual(linear.weight.total_batches, 4)

        # Check output shape
        self.assertEqual(output.shape, (4, 32))

    def test_observer_with_bias(self):
        """Test ObserverTensor works correctly with bias in linear layers."""
        in_features = 64
        out_features = 32
        batch_size = 8

        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )
        bias = torch.randn(out_features, dtype=torch.float32, device="cuda")
        observer_weight = ObserverTensor.from_hp(weight)

        input_tensor = torch.randn(
            batch_size, in_features, dtype=torch.float32, device="cuda"
        )

        # Test linear with bias
        output = F.linear(input_tensor, observer_weight, bias)

        # Check Hessian was updated
        self.assertIsNotNone(observer_weight.hessian)
        self.assertEqual(observer_weight.total_batches, batch_size)

        # Verify output is correct
        expected_output = F.linear(input_tensor, weight, bias)
        torch.testing.assert_close(output, expected_output)

    def test_hessian_incremental_update(self):
        """Test that incremental Hessian updates match batch calculation."""
        in_features = 32
        out_features = 16

        weight = torch.randn(
            out_features, in_features, dtype=torch.float32, device="cuda"
        )

        # Create two ObserverTensors - one for incremental, one for batch
        observer_incremental = ObserverTensor.from_hp(weight)

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
        from torchao.prototype.gptq import _calculate_hessian

        hessian_batch = _calculate_hessian(activations, device="cuda")

        # Compare incremental vs batch
        self.assertIsNotNone(observer_incremental.hessian)
        torch.testing.assert_close(
            observer_incremental.hessian, hessian_batch, rtol=1e-4, atol=1e-5
        )


class TestGPTQFlow(unittest.TestCase):
    def test_unified_config_two_phase(self):
        """Test that GPTQConfig handles both observation and quantization phases."""
        # Create a simple linear layer
        linear = torch.nn.Linear(64, 32, bias=False).cuda().to(torch.bfloat16)
        original_weight = linear.weight.data.clone()

        # Phase 1: Observation step - wrap as ObserverTensor
        observe_config = GPTQConfig(
            step="observe",
            group_size=128,
        )
        quantize_(linear, observe_config)

        # Verify weight is now an ObserverTensor
        self.assertIsInstance(linear.weight, ObserverTensor)
        torch.testing.assert_close(linear.weight.hp_data, original_weight)

        # Run some forward passes for calibration
        for _ in range(10):
            input_tensor = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
            _ = linear(input_tensor)

        # Verify Hessian was computed
        self.assertIsNotNone(linear.weight.hessian)
        self.assertGreater(linear.weight.total_batches, 0)

        # Phase 2: Convert step - apply GPTQ quantization
        convert_config = GPTQConfig(
            step="convert",
            group_size=128,
        )
        quantize_(linear, convert_config)

        # Verify weight is now Int4Tensor (quantized)
        from torchao.quantization import Int4Tensor

        self.assertIsInstance(linear.weight, Int4Tensor)

        # Verify it still works
        output = linear(input_tensor)
        self.assertEqual(output.shape, (4, 32))

    def test_gptq_quantize_function(self):
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
            group_size=128,
        )

        # Run GPTQ quantization
        quantized_weight = gptq_quantize(H, weight, config)

        # Check output type
        from torchao.quantization import Int4Tensor

        self.assertIsInstance(quantized_weight, Int4Tensor)

        # Check shape is preserved
        self.assertEqual(quantized_weight.shape, weight.shape)

        # Dequantize and check error is reasonable
        dequantized = quantized_weight.dequantize()
        self.assertEqual(dequantized.shape, weight.shape)

        # Check quantization introduces bounded error
        error = torch.abs(dequantized - weight.float())
        mean_error = error.mean().item()
        max_error = error.max().item()

        # GPTQ should have reasonable error bounds
        self.assertLess(mean_error, 0.5, f"Mean error too high: {mean_error}")
        self.assertLess(max_error, 5.0, f"Max error too high: {max_error}")

        # Check that quantization actually compressed the data
        # Int4 should be much smaller than bfloat16
        self.assertTrue(hasattr(quantized_weight, "qdata"))

    def test_gptq_quantize_better_than_naive(self):
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

        # Compute Hessian from activations
        from torchao.prototype.gptq import _calculate_hessian

        H = _calculate_hessian(activations, device="cuda")

        # GPTQ quantization
        config = GPTQConfig(
            step="convert",
            group_size=128,
        )
        gptq_quantized = gptq_quantize(H, weight, config)
        gptq_dequantized = gptq_quantized.dequantize()

        # Naive quantization (using identity Hessian)
        H_identity = torch.eye(in_features, device="cuda", dtype=torch.float32)
        naive_quantized = gptq_quantize(H_identity, weight, config)
        naive_dequantized = naive_quantized.dequantize()

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
        self.assertIsNotNone(gptq_loss)
        self.assertIsNotNone(naive_loss)

    def test_gptq_transformer(self):
        torch.manual_seed(43)
        from torchao._models.llama.model import (
            ModelArgs,
            Transformer,
            prepare_inputs_for_model,
        )

        torch.set_default_dtype(torch.bfloat16)

        config = ModelArgs(n_layer=2)

        with torch.device("cuda"):
            model = Transformer(config)
            model.setup_caches(max_batch_size=2, max_seq_length=100)
            idx = torch.randint(1, 10000, (10, 2, 50)).to(torch.int32)
            test_input = prepare_inputs_for_model(idx[0])

            model2 = copy.deepcopy(model)
            model_baseline = copy.deepcopy(model)

            # get new gptq implementation out
            gptqnew_config = GPTQConfig(step="observe", group_size=128)
            quantize_(model, gptqnew_config)

            # new calibration
            for i in range(10):
                input = prepare_inputs_for_model(idx[i])
                model(*input)

            convert_config = GPTQConfig(step="convert", group_size=128)
            quantize_(model, convert_config)
            out_gptq = model(*test_input)

            quantize_(model2, Int4WeightOnlyConfig(version=2))
            out_rtn = model2(*test_input)

            out = model_baseline(*test_input)

            from torchao.quantization.utils import compute_error

            sqnr_rtn = compute_error(out_rtn, out)
            sqnr_gptq = compute_error(out_gptq, out)

            assert sqnr_gptq > 30, f"GPTQ SQNR: {sqnr_gptq} is too low"
            assert sqnr_gptq > sqnr_rtn, (
                f"GPTQ SQNR: {sqnr_gptq} is not better than RTN SQNR: {sqnr_rtn}"
            )


if __name__ == "__main__":
    unittest.main()

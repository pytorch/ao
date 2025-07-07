# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import torch.nn.functional as F
from torchao.prototype.awq import (
    insert_awq_observer_qdq_,
    AWQQDQConfig,
)
from torchao.prototype.awq.executorch_awq import _is_awq_observed_linear_qdq
from torchao.quantization import quantize_
from torchao.dtypes.uintx.q_dq_layout import QDQLayout


class TestAWQExecutorchIntegration(unittest.TestCase):
    """Test suite for AWQ + QDQLayout + ExecuTorch integration."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

        # Create a simple test model
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 32),
        )

        # Example input for testing
        self.example_input = torch.randn(2, 16, 64)
        self.batch_size, self.seq_len, self.hidden_size = self.example_input.shape

    def test_awq_observer_insertion(self):
        """Test insertion of AWQ observers with QDQLayout support."""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 32),
        )

        # Insert AWQ observers
        insert_awq_observer_qdq_(
            model,
            n_validation_examples=2,
            validation_sequence_len=16,
            quant_dtype=torch.int4,
            group_size=64,
        )

        # Check that Linear layers were replaced with AWQObservedLinearQDQ
        from torchao.prototype.awq.executorch_awq import AWQObservedLinearQDQ

        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                # Should be replaced with AWQObservedLinearQDQ
                self.assertIsInstance(module, AWQObservedLinearQDQ)
                # Check observer configuration
                self.assertEqual(module.act_obs.n_validation_examples, 2)
                self.assertEqual(module.act_obs.validation_sequence_len, 16)

    def test_awq_calibration_and_quantization(self):
        """Test AWQ calibration and quantization with QDQLayout."""
        model = torch.nn.Sequential(torch.nn.Linear(64, 128))

        # Insert AWQ observer
        insert_awq_observer_qdq_(
            model,
            n_validation_examples=3,
            validation_sequence_len=16,
            quant_dtype=torch.int4,
            group_size=32,
        )

        # Calibrate the model
        model.eval()
        with torch.no_grad():
            for _ in range(3):
                example_input = torch.randn(2, 16, 64)
                model(example_input)

        # Apply quantization
        config = AWQQDQConfig(
            quant_dtype=torch.int4,
            group_size=32,
        )
        quantize_(model, config, filter_fn=_is_awq_observed_linear_qdq)

        # Verify the model is quantized (model is modified in-place)
        self.assertIsInstance(model, torch.nn.Sequential)
        self.assertIsInstance(model[0], torch.nn.Linear)

        # Check that weight uses QDQLayout
        weight_tensor = model[0].weight
        self.assertTrue(hasattr(weight_tensor, "__tensor_flatten__"))  # AQT tensor

        # Test forward pass
        with torch.no_grad():
            output = model(self.example_input)
            self.assertEqual(output.shape, (2, 16, 128))

    def test_multiple_quantization_dtypes(self):
        """Test AWQ with different quantization dtypes."""
        for quant_dtype in [torch.uint1, torch.uint2, torch.int4]:
            with self.subTest(quant_dtype=quant_dtype):
                model = torch.nn.Sequential(torch.nn.Linear(32, 64))

                # Insert observer
                insert_awq_observer_qdq_(
                    model,
                    n_validation_examples=2,
                    validation_sequence_len=4,
                    quant_dtype=quant_dtype,
                    group_size=16,
                )

                # Calibrate
                model.eval()
                with torch.no_grad():
                    for _ in range(2):
                        model(torch.randn(1, 4, 32))

                # Quantize
                config = AWQQDQConfig(quant_dtype=quant_dtype, group_size=16)
                quantize_(model, config, filter_fn=_is_awq_observed_linear_qdq)

                # Test forward pass
                with torch.no_grad():
                    output = model(torch.randn(1, 4, 32))
                    self.assertEqual(output.shape, (1, 4, 64))

    def test_different_group_sizes(self):
        """Test AWQ with different group sizes."""
        for group_size in [16, 32, 64, 128]:
            with self.subTest(group_size=group_size):
                model = torch.nn.Sequential(torch.nn.Linear(128, 64))

                # Insert observer
                insert_awq_observer_qdq_(
                    model,
                    n_validation_examples=2,
                    validation_sequence_len=4,
                    quant_dtype=torch.int4,
                    group_size=group_size,
                )

                # Calibrate
                model.eval()
                with torch.no_grad():
                    for _ in range(2):
                        model(torch.randn(1, 4, 128))

                # Quantize
                config = AWQQDQConfig(quant_dtype=torch.int4, group_size=group_size)
                quantize_(model, config, filter_fn=_is_awq_observed_linear_qdq)

                # Test forward pass
                with torch.no_grad():
                    output = model(torch.randn(1, 4, 128))
                    self.assertEqual(output.shape, (1, 4, 64))

    def test_graph_pattern_for_executorch(self):
        """Test that the graph pattern matches ExecuTorch expectations for XNNPACK lowering."""
        model = torch.nn.Sequential(torch.nn.Linear(128, 64))

        # Insert AWQ observers with dynamic activation quantization
        insert_awq_observer_qdq_(
            model,
            n_validation_examples=2,
            validation_sequence_len=8,
            quant_dtype=torch.int4,
            group_size=32,
        )

        # Calibrate
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                model(torch.randn(1, 8, 128))

        # Quantize
        config = AWQQDQConfig(
            quant_dtype=torch.int4,
            group_size=32,
        )
        quantize_(model, config, filter_fn=_is_awq_observed_linear_qdq)

        # Test the forward method applies the expected AWQ + dynamic activation quantization pattern
        example_input = torch.randn(1, 8, 128)

        # Test that forward pass runs without error
        with torch.no_grad():
            actual_output = model(example_input)

        # Verify output shape is correct
        self.assertEqual(actual_output.shape, (1, 8, 64))

        # Test graph pattern using torch.export (the proper way for ExecuTorch)
        # Export with strict=True for ExecuTorch compatibility
        exported_program = torch.export.export(model, (example_input,), strict=True)

        # Test that exported model produces same results
        exported_results = exported_program.module()(example_input)
        self.assertTrue(
            torch.allclose(actual_output, exported_results, atol=1e-3),
            "Exported model should produce same results as original",
        )

        # Use FileCheck to verify the graph contains required operations for AWQ + dynamic activation quantization
        from torch.testing import FileCheck

        # Expected operations in the exported graph for AWQ + dynamic activation quantization
        # This pattern is what ExecuTorch can recognize and lower to XNNPACK:
        # 1. AWQ scaling (division operation)
        # 2. Dynamic activation quantization (choose_qparams, quantize, dequantize)
        # 3. Weight quantization/dequantization (from QDQLayout)
        # 4. Linear operation on dequantized tensors
        expected_operations = [
            # AWQ scaling - division operation to scale input by AWQ scale
            "torch.ops.aten.div.Tensor",
            # Dynamic activation quantization - choose quantization parameters
            "torch.ops.torchao.choose_qparams_affine.default",
            # Dynamic activation quantization - quantize activation
            "torch.ops.torchao.quantize_affine.default",
            # Dynamic activation dequantization - dequantize activation for linear op
            "torch.ops.torchao.dequantize_affine.default",
            # Linear operation on dequantized tensors
            "torch.ops.aten.linear.default",
        ]

        # Verify each required operation appears in the exported graph
        for operation in expected_operations:
            count = 1
            # We expect 2 dequantize operations: one for activation, one for weight
            if operation == "torch.ops.torchao.dequantize_affine.default":
                count = 2
            FileCheck().check_count(operation, count, exactly=True).run(
                exported_program.graph_module.code
            )


if __name__ == "__main__":
    unittest.main()

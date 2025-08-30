# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from copy import deepcopy

import torch
from torch.testing._internal import common_utils

from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
    SmoothQuantObservedLinear,
)
from torchao.prototype.smoothquant.core import SmoothQuantStep
from torchao.quantization import quantize_
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
)


@unittest.skipIf(torch.version.hip is not None, "Skipping tests in ROCm")
class TestSmoothQuant(unittest.TestCase):
    """SmoothQuant tests using only supported quantization configs."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level configuration for tests."""
        # This test case will trigger recompilation many times, so set a large cache_size_limit here
        torch._dynamo.config.cache_size_limit = 128

    # TODO: Update after #2729 merged
    # from torchao.testing.model_architectures import ToyMultiLinearModel
    class ToyMultiLinearModel(torch.nn.Module):
        """Shared model class for testing"""

        def __init__(self, m=512, n=256, k=128, has_bias=False):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, n, bias=has_bias)
            self.linear2 = torch.nn.Linear(n, k, bias=has_bias)
            self.linear3 = torch.nn.Linear(k, 64, bias=has_bias)

        def example_inputs(
            self, batch_size=1, sequence_length=10, dtype=torch.float, device="cuda"
        ):
            return [
                torch.randn(
                    1,
                    sequence_length,
                    self.linear1.in_features,
                    dtype=dtype,
                    device=device,
                )
                for _ in range(batch_size)
            ]

        def forward(self, x):
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            return x

    @common_utils.parametrize("alpha", [None, 0.5, 0.75])
    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            # Note: float8_static_activation_float8_weight is broken after recent PyTorch update.
            # TODO(#1639): Fix for supporting more API in torchao/quantization/quant_api.py
        ],
    )
    @common_utils.parametrize("device", ["cpu", "cuda"])
    @common_utils.parametrize("input_dtype", [torch.float, torch.bfloat16, torch.half])
    def test_smoothquant_accuracy(self, alpha, base_config, device, input_dtype):
        """Test the margin error of SmoothQuant across bias, alpha, dtype, etc."""

        m = self.ToyMultiLinearModel(32, 16, 8).eval().to(device).to(input_dtype)
        m_ref = deepcopy(m)
        test_data = torch.randn(32, 32, dtype=input_dtype, device=device)

        # Step 1: Setup quantized model with observer insertion and calibration
        config = SmoothQuantConfig(
            base_config=base_config,
            step=SmoothQuantStep.PREPARE,
            alpha=alpha,
        )
        quantize_(m, config)

        # Perform calibration with test data
        m(test_data)

        # Apply quantization configuration
        config.step = SmoothQuantStep.CONVERT
        quantize_(m, config)

        # Step 2: Inference quantized model
        with torch.inference_mode():
            q_out = m(test_data)
            ref_out = m_ref(test_data)

            # Simple validation - ensure quantized model produces reasonable outputs
            self.assertIsNotNone(q_out, "Quantized model output should not be None")
            self.assertFalse(
                torch.isnan(q_out).any(),
                f"Quantized model should not produce NaN values for "
                f"alpha={alpha}, base_config={type(base_config).__name__}, "
                f"device={device}, dtype={input_dtype}",
            )

            # Check output shapes match
            self.assertEqual(
                q_out.shape,
                ref_out.shape,
                f"Output shapes should match: quantized={q_out.shape}, reference={ref_out.shape}",
            )

    def test_observer_insertion(self):
        """Test that PREPARE step correctly inserts SmoothQuantObservedLinear."""

        m = self.ToyMultiLinearModel(has_bias=False).eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE step - should insert observers
        config = SmoothQuantConfig(
            base_config=Int8DynamicActivationInt8WeightConfig(),
            step=SmoothQuantStep.PREPARE,
        )
        quantize_(m, config)

        # After PREPARE - should be SmoothQuantObservedLinear
        self.assertIsInstance(m.linear1, SmoothQuantObservedLinear)
        self.assertTrue(hasattr(m.linear1, "obs"))

        # Test calibration
        test_data = torch.randn(2, 512)
        m(test_data)

        # CONVERT step - should produce regular Linear with quantized weights
        config.step = SmoothQuantStep.CONVERT
        quantize_(m, config)

        # After CONVERT - should be regular Linear again (but quantized)
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

    def test_prepare_for_loading(self):
        """Test PREPARE_FOR_LOADING step for loading pre-quantized checkpoints."""

        m = self.ToyMultiLinearModel(has_bias=False).eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE_FOR_LOADING step - should create quantized model ready for loading
        config = SmoothQuantConfig(
            base_config=Int8DynamicActivationInt8WeightConfig(),
            step=SmoothQuantStep.PREPARE_FOR_LOADING,
            alpha=0.5,
        )
        quantize_(m, config)

        # After PREPARE_FOR_LOADING - should be regular Linear with quantized weights
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # Test that model can run inference
        test_data = torch.randn(2, 512)
        with torch.inference_mode():
            output = m(test_data)

            # Validate output
            self.assertIsNotNone(
                output, "PREPARE_FOR_LOADING model output should not be None"
            )
            self.assertFalse(
                torch.isnan(output).any(), "Model should not produce NaN values"
            )
            self.assertEqual(
                output.shape, (2, 64), "Output shape should match expected dimensions"
            )

    # TODO: Check more quantization APIs and dtype
    @common_utils.parametrize("alpha", [None, 0.5, 0.75])
    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            # Skip int4 weight tests for now due to reference implementation mismatch
            # Int8DynamicActivationInt4WeightConfig(),
        ],
    )
    @common_utils.parametrize("device", ["cpu", "cuda"])
    @common_utils.parametrize("input_dtype", [torch.float])
    def test_two_step_quantization(self, alpha, base_config, device, input_dtype):
        """Test two-step quantization process (PREPARE -> CONVERT)."""
        dataset_size = 20
        n_calib_examples = 10
        sequence_length = 20  # Must be > 16 to avoid CUDA int_mm limitation

        # Create model and move to device/dtype
        m1 = (
            self.ToyMultiLinearModel(512, 256, 128, has_bias=False)
            .eval()
            .to(device)
            .to(input_dtype)
        )
        m2 = deepcopy(m1)

        # Generate calibration dataset
        dataset = m1.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=input_dtype,
            device=device,
        )
        calibration_data = dataset[:n_calib_examples]

        # Step 1: PREPARE - Insert observers
        config = SmoothQuantConfig(
            base_config=base_config, step=SmoothQuantStep.PREPARE, alpha=alpha
        )
        quantize_(m2, config)

        # Step 2: Calibration
        for data in calibration_data:
            m2(data.squeeze(0).to(input_dtype))

        # Step 3: Apply quantization configuration
        config.step = SmoothQuantStep.CONVERT
        quantize_(m2, config)

        # Step 4: Validate outputs on full dataset
        with torch.inference_mode():
            m2_outputs = []

            for data in dataset:
                # TODO: Remove fixed dtype for testing more quantization APIs
                input_tensor = data.squeeze(0).float()
                m2_output = m2(input_tensor)
                m2_outputs.append(m2_output)

            # Concatenate all outputs
            m2_result = torch.cat(m2_outputs)

            self.assertIsNotNone(m2_result, "Quantized model output should not be None")

            # Check that model produces reasonable outputs
            self.assertFalse(
                torch.isnan(m2_result).any(),
                f"Quantized model should not produce NaN values for "
                f"alpha={alpha}, base_config={type(base_config).__name__}, device={device}, dtype={input_dtype}",
            )


common_utils.instantiate_parametrized_tests(TestSmoothQuant)

if __name__ == "__main__":
    unittest.main()

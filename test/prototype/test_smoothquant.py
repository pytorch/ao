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

    from test.prototype.test_awq import ToyLinearModel

    @common_utils.parametrize("alpha", [0.5, 0.75])
    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            # Note: float8_static_activation_float8_weight is broken after recent PyTorch update.
            # TODO(#1639): Fix for supporting more API in torchao/quantization/quant_api.py
        ],
    )
    @common_utils.parametrize("device", ["cpu", "cuda"])
    @common_utils.parametrize("input_dtype", [torch.bfloat16])
    # TODO: Replace ToyLinearModel with Transformer for real run.
    # For verifying SmoothQuant, we have to compare if accuracy with SmoothQuant is higher than basic quantization
    # See https://github.com/pytorch/ao/pull/2728#discussion_r2319143330 for more info
    def test_smoothquant_accuracy(self, alpha, base_config, device, input_dtype):
        """Test the margin error of SmoothQuant across bias, alpha, dtype, etc."""

        m = self.ToyLinearModel(32, 16, 8).eval().to(device).to(input_dtype)
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

    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            # TODO: Check more quantization APIs
        ],
    )
    def test_observer_insertion(self, base_config):
        """Test that PREPARE step correctly inserts SmoothQuantObservedLinear."""

        m = self.ToyLinearModel().eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE step - should insert observers
        config = SmoothQuantConfig(
            base_config=base_config,
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

    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            # TODO: Check more quantization APIs
        ],
    )
    def test_prepare_for_loading(self, base_config):
        """Test PREPARE_FOR_LOADING step for loading pre-quantized checkpoints."""

        m = self.ToyLinearModel().eval()

        # Before quantization - should be regular Linear
        self.assertIsInstance(m.linear1, torch.nn.Linear)
        self.assertNotIsInstance(m.linear1, SmoothQuantObservedLinear)

        # PREPARE_FOR_LOADING step - should create quantized model ready for loading
        config = SmoothQuantConfig(
            base_config=base_config,
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


common_utils.instantiate_parametrized_tests(TestSmoothQuant)

if __name__ == "__main__":
    unittest.main()

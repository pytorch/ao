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


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 64, bias=False)

    def example_inputs(
        self,
        batch_size,
        sequence_length=10,
        dtype=torch.bfloat16,
        device="cuda",
    ):
        return [
            torch.randn(
                1,
                sequence_length,
                self.linear1.in_features,
                dtype=dtype,
                device=device,
            )
            for j in range(batch_size)
        ]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(torch.version.hip is not None, "Skipping tests in ROCm")
class TestSmoothQuant(unittest.TestCase):
    """SmoothQuant tests using only supported quantization configs."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level configuration for tests."""
        # This test case will trigger recompilation many times, so set a large cache_size_limit here
        torch._dynamo.config.cache_size_limit = 128

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
    def test_smoothquant_accuracy(self, alpha, base_config, device, input_dtype):
        """Test if SmoothQuant achieves lower loss than basic quantization."""
        in_features = 64
        out_features = 128

        # Note: This is sanity check. For real run, consider Transformer model to reproduce.
        X = torch.randn(16, in_features, dtype=input_dtype, device=device)
        W = torch.randn(out_features, in_features, dtype=input_dtype, device=device)

        # Create linear layer
        linear = (
            torch.nn.Linear(in_features, out_features, bias=False)
            .to(device)
            .to(input_dtype)
        )
        with torch.no_grad():
            linear.weight.copy_(W)

        # Reference output
        out_ref = linear(X)

        # Step 1. Basic quantization
        basic_model = deepcopy(linear)
        quantize_(basic_model, base_config)
        out_basic = basic_model(X)
        loss_base = torch.nn.functional.mse_loss(out_basic, out_ref).item()

        # SmoothQuant quantization
        model = deepcopy(linear)
        config = SmoothQuantConfig(
            base_config=base_config,
            step=SmoothQuantStep.PREPARE,
            alpha=alpha,
        )
        quantize_(model, config)

        # Perform calibration with test data
        model(X)

        # Step 2. SmoothQuant
        config.step = SmoothQuantStep.CONVERT
        quantize_(model, config)

        out_smoothquant = model(X)
        loss_smoothquant = torch.nn.functional.mse_loss(out_smoothquant, out_ref).item()

        assert loss_smoothquant < loss_base, (
            f"SmoothQuant loss ({loss_smoothquant:.6f}) should not be higher than basic loss ({loss_base:.6f})"
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

        m = ToyLinearModel().eval()

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

        m = ToyLinearModel().eval()

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

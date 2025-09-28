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
from torchao.quantization.linear_activation_scale import (
    WeightTensorWithLinearActivationScaleMetadata,
)
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
    Int8StaticActivationInt8WeightConfig,
)
from torchao.quantization.utils import (
    compute_error as SQNR,
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
        return (
            torch.randn(
                batch_size,
                sequence_length,
                self.linear1.in_features,
                dtype=dtype,
                device=device,
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]


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
    @common_utils.parametrize("device", device_list)
    @common_utils.parametrize("input_dtype", [torch.bfloat16])
    def test_smoothquant_accuracy(self, alpha, base_config, device, input_dtype):
        """Test if SmoothQuant achieves lower loss than basic quantization."""
        torch.manual_seed(42)
        in_features = 64
        out_features = 128

        class Mod(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = (
                    torch.nn.Linear(in_features, out_features, bias=False)
                    .to(device)
                    .to(input_dtype)
                )

            def forward(self, x):
                return self.linear(x)

        # Note: This is sanity check. For real run, consider Transformer model to reproduce.
        X = torch.randn(16, in_features, dtype=input_dtype, device=device)
        W = torch.randn(out_features, in_features, dtype=input_dtype, device=device)

        # Create linear layer
        m = Mod().eval()
        with torch.no_grad():
            m.linear.weight.copy_(W)

        # Reference output
        out_ref = m(X)

        # Step 1. Basic quantization
        basic_model = deepcopy(m)
        quantize_(basic_model, base_config)
        out_basic = basic_model(X)
        loss_base = torch.nn.functional.mse_loss(out_basic, out_ref).item()

        # SmoothQuant quantization
        model = deepcopy(m)
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
        assert isinstance(
            model.linear.weight, WeightTensorWithLinearActivationScaleMetadata
        )

        out_smoothquant = model(X)
        loss_smoothquant = torch.nn.functional.mse_loss(out_smoothquant, out_ref).item()

        assert loss_smoothquant < loss_base, (
            f"SmoothQuant loss ({loss_smoothquant:.6f}) should not be higher than basic loss ({loss_base:.6f})"
        )

    @common_utils.parametrize("alpha", [0.5, 0.25])
    @common_utils.parametrize("device", ["cpu"])
    @common_utils.parametrize("input_dtype", [torch.bfloat16])
    def test_smoothquant_static_act_accuracy(self, alpha, device, input_dtype):
        """Test if SmoothQuant with static quantization achieves lower loss than basic quantization."""
        torch.manual_seed(42)
        m = ToyLinearModel().eval().to(device).to(input_dtype)
        x = m.example_inputs(batch_size=1, dtype=input_dtype, device=device)

        # Output without quantization
        y_nq = m(*x)

        # Step 1. Reference with alpha=0
        m_ref = deepcopy(m)
        base_config = Int8StaticActivationInt8WeightConfig()
        config = SmoothQuantConfig(
            base_config=base_config,
            step=SmoothQuantStep.PREPARE,
            alpha=0.0,
        )
        with torch.no_grad():
            quantize_(m_ref, config)
            # Perform calibration with test data
            m_ref(*x)
            # Step 2. SmoothQuant
            config.step = SmoothQuantStep.CONVERT
            quantize_(m_ref, config)
            y_ref = m_ref(*x)
        loss_base = torch.nn.functional.mse_loss(y_ref, y_nq).item()

        # SmoothQuant quantization
        base_config = Int8StaticActivationInt8WeightConfig()
        config = SmoothQuantConfig(
            base_config=base_config,
            step=SmoothQuantStep.PREPARE,
            alpha=alpha,
        )
        with torch.no_grad():
            quantize_(m, config)

            # Perform calibration with test data
            m(*x)

            # Step 2. SmoothQuant
            config.step = SmoothQuantStep.CONVERT
            quantize_(m, config)
            y = m(*x)
        assert isinstance(
            m.linear1.weight, WeightTensorWithLinearActivationScaleMetadata
        )
        assert isinstance(
            m.linear2.weight, WeightTensorWithLinearActivationScaleMetadata
        )
        loss_smoothquant = torch.nn.functional.mse_loss(y, y_nq).item()

        assert loss_smoothquant < loss_base, (
            f"SmoothQuant loss ({loss_smoothquant:.6f}) should not be higher than basic loss ({loss_base:.6f})"
        )
        sqnr = SQNR(y_ref, y)
        self.assertGreater(sqnr, 20.0)

    @common_utils.parametrize(
        "base_config",
        [
            Int8DynamicActivationInt8WeightConfig(),
            Int8StaticActivationInt8WeightConfig(),
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
            Int8StaticActivationInt8WeightConfig(),
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

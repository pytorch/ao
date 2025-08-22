# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest
from copy import deepcopy

import torch
from torch.testing._internal import common_utils

from torchao.prototype.smoothquant import (
    SmoothQuantConfig,
    SmoothQuantObservedLinear,
    insert_smooth_quant_observer_,
    load_smooth_quant_recipe,
    save_smooth_quant_recipe,
)
from torchao.quantization import quantize_
from torchao.quantization.utils import (
    dequantize_per_channel,
    dynamically_quantize_per_channel,
)
from torchao.testing.model_architectures import ToyTwoLinearModel


@unittest.skipIf(torch.version.hip is not None, "Skipping tests in ROCm")
class TestSmoothQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class-level configuration for tests."""
        # This test case will trigger recompilation many times, so set a large cache_size_limit here
        torch._dynamo.config.cache_size_limit = 128

    @unittest.skip("This test is broken on recent PyTorch, TODO(#1639): fix it")
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("alpha", [None, 0.5, 0.75])
    @common_utils.parametrize("quant_mode", ["static", "dynamic"])
    @common_utils.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    @common_utils.parametrize("input_dtype", [torch.float, torch.bfloat16, torch.half])
    def test_smoothquant_accuracy(self, bias, alpha, quant_mode, device, input_dtype):
        """Test the margin error of SmoothQuant across bias, alpha, dtype, etc."""

        class SimpleLinear(torch.nn.Module):
            def __init__(self, bias: bool):
                super().__init__()
                self.fc = torch.nn.Linear(32, 32, bias)
                self.fc.weight.data = torch.randn_like(self.fc.weight.data)

            def forward(self, x):
                return self.fc(x)

        # Create model, reference, and test data
        m = SimpleLinear(bias).eval().to(input_dtype).to(device)
        m_ref = deepcopy(m)
        test_data = torch.randn(2, 32, dtype=input_dtype, device=device)

        # Step 1: Setup quantized model with observer insertion and calibration
        insert_smooth_quant_observer_(m, alpha, quant_mode)

        # Perform calibration with test data
        m(test_data)

        # Apply quantization configuration
        is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
        quantize_(m, SmoothQuantConfig(), is_observed_linear)

        # Apply compilation if supported
        m = torch.compile(m, fullgraph=True)

        # Step 2: Inference quantized model
        with torch.inference_mode():
            q_out = m(test_data)

            # Step 3: Compute reference
            weight = m_ref.fc.weight.data.float()
            b = m_ref.fc.bias if bias else None
            x_abs_max_per_ic = torch.abs(test_data).max(dim=0).values
            w_abs_max_per_ic = torch.abs(weight).max(dim=0).values

            if alpha is not None:
                # Apply SmoothQuant
                smoothing_factor = torch.pow(x_abs_max_per_ic, alpha) / torch.pow(
                    w_abs_max_per_ic, 1 - alpha
                )
            else:
                smoothing_factor = torch.ones_like(x_abs_max_per_ic)

            # Apply smoothing to activations and weights
            smoothed_activation = test_data / smoothing_factor
            smoothed_weight = weight * smoothing_factor

            # Quantize weights using per-channel quantization
            qw, w_scales, w_zps = dynamically_quantize_per_channel(
                smoothed_weight, -127, 127, torch.int8
            )
            fq_wei = dequantize_per_channel(qw, w_scales, w_zps, input_dtype)

            # Handle activation quantization based on mode
            if quant_mode == "static":
                # activation is quantized per-tensor
                act_min, act_max = torch.aminmax(smoothed_activation.float())
                max_val_pos = torch.max(-act_min, act_max)
                activation_scale = max_val_pos / 127.0

                fq_act = (
                    torch.quantize_per_tensor(
                        smoothed_activation.float(),
                        scale=activation_scale.item(),
                        zero_point=0,
                        dtype=torch.qint8,
                    )
                    .dequantize()
                    .to(input_dtype)
                )
            else:
                # activation is quantized per-row (batch * sequence_length)
                qx, x_scales, x_zps = dynamically_quantize_per_channel(
                    smoothed_activation.float(), -127, 127, torch.int8
                )
                fq_act = dequantize_per_channel(
                    qx,
                    x_scales,
                    x_zps,
                    input_dtype,
                )

            # Compute final linear operation
            reference_out = torch.nn.functional.linear(fq_act, fq_wei, b)

            # Step 4: Validate numerical accuracy
            tolerance = (
                0.1
                if input_dtype == torch.float
                else (0.2 if input_dtype == torch.half else 0.3)
            )
            torch.testing.assert_close(
                q_out,
                reference_out.to(input_dtype),
                atol=tolerance,
                msg=f"Quantized output differs from reference for "
                f"bias={bias}, alpha={alpha}, quant_mode={quant_mode}, "
                f"device={device}, dtype={input_dtype}",
            )

    @unittest.skip("This test is broken on recent PyTorch, TODO(#1639): fix it")
    @common_utils.parametrize("alpha", [None, 0.5, 0.75])
    @common_utils.parametrize("quant_mode", ["static", "dynamic"])
    @common_utils.parametrize(
        "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )
    @common_utils.parametrize("input_dtype", [torch.float, torch.bfloat16, torch.half])
    def test_save_load_recipe(self, alpha, quant_mode, device, input_dtype):
        """Test save/load recipe functionality."""
        dataset_size = 20
        layer_dims = (512, 256, 128)  # Input, hidden, output dimensions
        n_calib_examples = 10
        sequence_length = 5

        # Create two identical models for comparison
        m = ToyTwoLinearModel(*layer_dims).eval().to(input_dtype).to(device)
        m_save_load = deepcopy(m)

        # Generate calibration dataset
        dataset = m.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=input_dtype,
            device=device,
        )
        calibration_data = dataset[:n_calib_examples]

        # Step 1: Setup first quantized model with observer insertion and calibration
        insert_smooth_quant_observer_(m, alpha, quant_mode)

        # Perform calibration with calibration data
        for data in calibration_data:
            m(data)

        # Apply quantization configuration
        is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
        quantize_(m, SmoothQuantConfig(), is_observed_linear)

        # Apply compilation if supported
        m = torch.compile(m, fullgraph=True)

        # Step 2: Setup save/load model with recipe functionality
        insert_smooth_quant_observer_(m_save_load, alpha, quant_mode)
        for example in calibration_data:
            m_save_load(example.to(device))

        # Step 3: Test save/load recipe functionality
        with tempfile.NamedTemporaryFile() as temp_file:
            save_path = temp_file.name
            save_smooth_quant_recipe(m_save_load, save_path)
            load_smooth_quant_recipe(m_save_load, save_path)

            # Step 4: Complete quantization for save/load model
            is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
            quantize_(m_save_load, SmoothQuantConfig(), is_observed_linear)

            m_save_load = torch.compile(m_save_load, fullgraph=True)

            # Step 5: Validate outputs on full dataset
            with torch.inference_mode():
                original_outputs = []
                save_load_outputs = []

                for data in dataset:
                    # Remove batch dimension for model input
                    input_tensor = data.squeeze(0)

                    original_output = m(input_tensor)
                    save_load_output = m_save_load(input_tensor)

                    original_outputs.append(original_output)
                    save_load_outputs.append(save_load_output)

                # Concatenate all outputs for comparison
                original_result = torch.cat(original_outputs)
                save_load_out = torch.cat(save_load_outputs)

                self.assertIsNotNone(
                    original_result, "Original model output should not be None"
                )
                self.assertIsNotNone(
                    save_load_out, "Save/load model output should not be None"
                )

                torch.testing.assert_close(
                    original_result,
                    save_load_out,
                    msg=f"Save/load recipe should produce identical results for "
                    f"alpha={alpha}, quant_mode={quant_mode}, device={device}, dtype={input_dtype}",
                )


common_utils.instantiate_parametrized_tests(TestSmoothQuant)

if __name__ == "__main__":
    unittest.main()

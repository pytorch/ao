# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import tempfile
import unittest
from copy import deepcopy

import torch

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
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=512, n=256, k=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)
        self.linear3 = torch.nn.Linear(k, 1, bias=False)

    def example_inputs(
        self, batch_size, sequence_length=10, dtype=torch.bfloat16, device="cuda"
    ):
        return [
            torch.randn(
                1, sequence_length, self.linear1.in_features, dtype=dtype, device=device
            )
            for j in range(batch_size)
        ]

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


class TestSmoothQuant(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up class-level configuration for tests."""
        # Skip tests on ROCm (AMD GPU) due to compatibility issues
        if torch.version.hip is not None:
            raise unittest.SkipTest("Skipping the tests in ROCm")

        if TORCH_VERSION_AT_LEAST_2_5:
            # This test case will trigger recompilation many times, so set a large cache_size_limit here
            torch._dynamo.config.cache_size_limit = 128

        # Define test parameter ranges
        cls.bias_options = [True, False]
        cls.alpha_options = [None, 0.5, 0.75]  # None means conventional quantization
        cls.quant_mode_options = ["static", "dynamic"]
        cls.devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        cls.input_dtypes = (torch.float, torch.bfloat16, torch.half)

    @unittest.skip("This test is broken on recent PyTorch, TODO(#1639): fix it")
    def test_smoothquant_accuracy(self):
        """Test the margin error of SmoothQuant across bias, alpha, dtype, etc."""

        # Test all parameter combinations using subTest for better isolation
        for bias in self.bias_options:
            for alpha in self.alpha_options:
                for quant_mode in self.quant_mode_options:
                    for device in self.devices:
                        for input_dtype in self.input_dtypes:
                            with self.subTest(
                                bias=bias,
                                alpha=alpha,
                                quant_mode=quant_mode,
                                device=device,
                                input_dtype=input_dtype,
                            ):
                                self._run_compute_accuracy_test(
                                    bias, alpha, quant_mode, device, input_dtype
                                )

    def _run_compute_accuracy_test(self, bias, alpha, quant_mode, device, input_dtype):
        """Single compute accuracy test"""

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

        # Step 1: Get calibration from observed SmoothQuant
        insert_smooth_quant_observer_(m, alpha, quant_mode)
        m(test_data)

        # Step 2: Quantize
        is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
        quantize_(m, SmoothQuantConfig(), is_observed_linear)

        # Step 3: Inference quantized model
        with torch.inference_mode():
            if TORCH_VERSION_AT_LEAST_2_5:
                m = torch.compile(m, fullgraph=True)
            q_out = m(test_data)

            # Step 4: Compute reference
            reference_out = self._compute_reference_out(
                m_ref, test_data, alpha, quant_mode, bias, input_dtype
            )

            # Step 5: Validate numerical accuracy
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

    def _compute_reference_out(self, m_ref, data, alpha, quant_mode, bias, input_dtype):
        """Compute the expected SmoothQuant output."""
        weight = m_ref.fc.weight.data.float()
        b = m_ref.fc.bias if bias else None
        x_abs_max_per_ic = torch.abs(data).max(dim=0).values
        w_abs_max_per_ic = torch.abs(weight).max(dim=0).values
        if alpha is not None:
            # Apply SmoothQuant
            smoothing_factor = torch.pow(x_abs_max_per_ic, alpha) / torch.pow(
                w_abs_max_per_ic, 1 - alpha
            )
        else:
            smoothing_factor = torch.ones_like(x_abs_max_per_ic)

        # Apply smoothing to activations and weights
        smoothed_activation = data / smoothing_factor
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
        return torch.nn.functional.linear(fq_act, fq_wei, b)

    @unittest.skip("This test is broken on recent PyTorch, TODO(#1639): fix it")
    def test_save_load_recipe(self):
        """Setup test for save/load recipe functionality."""
        for alpha in self.alpha_options:
            for quant_mode in self.quant_mode_options:
                for device in self.devices:
                    for input_dtype in self.input_dtypes:
                        with self.subTest(
                            alpha=alpha,
                            quant_mode=quant_mode,
                            device=device,
                            input_dtype=input_dtype,
                        ):
                            self._run_save_load_recipe_test(
                                alpha, quant_mode, device, input_dtype
                            )

    def _run_save_load_recipe_test(self, alpha, quant_mode, device, input_dtype):
        """Single save/load recipe test."""
        dataset_size = 20
        layer_dims = (512, 256, 128)  # Input, hidden, output dimensions
        n_calib_examples = 10
        sequence_length = 5

        # Create two identical models for comparison
        m = ToyLinearModel(*layer_dims).eval().to(input_dtype).to(device)
        m_save_load = deepcopy(m)

        # Generate calibration dataset
        dataset = m.example_inputs(
            dataset_size,
            sequence_length=sequence_length,
            dtype=input_dtype,
            device=device,
        )
        calibration_data = dataset[:n_calib_examples]

        # Step 1: Insert observers in both models
        insert_smooth_quant_observer_(m, alpha, quant_mode)
        insert_smooth_quant_observer_(m_save_load, alpha, quant_mode)

        # Step 2: Calibrate both models with identical data
        for example in calibration_data:
            m(example.to(device))
            m_save_load(example.to(device))

        # Step 3: Test save/load recipe functionality
        with tempfile.NamedTemporaryFile() as temp_file:
            save_path = temp_file.name
            save_smooth_quant_recipe(m_save_load, save_path)
            load_smooth_quant_recipe(m_save_load, save_path)

            # Step 4: Quantize both models
            is_observed_linear = lambda m, fqn: isinstance(m, SmoothQuantObservedLinear)
            quantize_(m, SmoothQuantConfig(), is_observed_linear)
            quantize_(m_save_load, SmoothQuantConfig(), is_observed_linear)

            if TORCH_VERSION_AT_LEAST_2_5:
                # earlier versions are not compatible
                m = torch.compile(m, fullgraph=True)
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


if __name__ == "__main__":
    unittest.main()

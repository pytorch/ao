# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import pytest

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import copy
import io
import random
import unittest
from contextlib import nullcontext
from functools import partial
from typing import Tuple

import pytest
import torch
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal import common_utils

from torchao.float8.float8_utils import compute_error
from torchao.quantization import (
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_api import (
    float8_static_activation_float8_weight,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
)
from torchao.utils import (
    is_sm_at_least_89,
    is_sm_at_least_90,
)

random.seed(0)
torch.manual_seed(0)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestAffineQuantizedFloat8Compile(InductorTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "granularity", [PerTensor(), PerRow()] if is_sm_at_least_90() else [PerTensor()]
    )
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 64, 256),
        ],
    )
    def test_fp8_linear_variants(
        self, dtype: torch.dtype, mode: str, compile: bool, sizes: Tuple, granularity
    ):
        error_message = None
        if isinstance(granularity, PerRow):
            if mode == "dynamic" and dtype != torch.bfloat16:
                error_message = "PerRow quantization only works for bfloat16 precision"
            elif mode == "static":
                error_message = (
                    "Static quantization only supports PerTensor granularity"
                )

        error_context = (
            pytest.raises(AssertionError, match=error_message)
            if error_message
            else nullcontext()
        )

        with error_context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")
            # Get a "reasonable" scale for the input tensor even though
            # we use the same scale for multiple activations
            scale, _ = choose_qparams_affine(
                input_tensor,
                MappingType.SYMMETRIC,
                input_tensor.shape,
                torch.float8_e4m3fn,
                scale_dtype=torch.float32,
            )
            mode_map = {
                "dynamic": partial(
                    float8_dynamic_activation_float8_weight, granularity=granularity
                ),
                "weight-only": float8_weight_only,
                "static": partial(
                    float8_static_activation_float8_weight,
                    scale=scale,
                    granularity=granularity,
                ),
            }

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)
            factory = mode_map[mode]()
            quantize_(quantized_model, factory)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert compute_error(output_original, output_quantized) > 20, (
                f"Quantization error is too high got a SQNR of {error}"
            )

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            float8_dynamic_activation_float8_weight(granularity="invalid")

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_mismatched_granularity(self):
        with pytest.raises(
            ValueError,
            match="Different granularities for activation and weight are not supported",
        ):
            float8_dynamic_activation_float8_weight(granularity=(PerTensor(), PerRow()))

    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_unsupported_granularity(self):
        class UnsupportedGranularity:
            pass

        with pytest.raises(ValueError, match="Invalid granularity types"):
            float8_dynamic_activation_float8_weight(
                granularity=(UnsupportedGranularity(), UnsupportedGranularity())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_per_row_with_float32(self):
        with pytest.raises(
            AssertionError,
            match="PerRow quantization only works for bfloat16 precision",
        ):
            model = ToyLinearModel(64, 64).eval().to(torch.float32).to("cuda")
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerRow())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    @common_utils.parametrize("mode", ["dynamic", "weight-only", "static"])
    def test_serialization(self, mode: str):
        # Create and quantize the model
        model = ToyLinearModel(16, 32).to(device="cuda")

        mode_map = {
            "dynamic": partial(
                float8_dynamic_activation_float8_weight, granularity=PerTensor()
            ),
            "weight-only": float8_weight_only,
            "static": partial(
                float8_static_activation_float8_weight,
                scale=torch.tensor(1.0, dtype=torch.float32, device="cuda"),
                granularity=PerTensor(),
            ),
        }
        factory = mode_map[mode]()
        quantize_(model, factory)

        # Save the state dict to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)

        # Reset the buffer position
        buffer.seek(0)

        # Load the state dict from the buffer
        weights_only_load = True
        loaded_state_dict = torch.load(buffer, weights_only=weights_only_load)

        # Create a new model and load the state dict
        with torch.device("meta"):
            new_model = ToyLinearModel(16, 32)
            if mode == "static":
                quantize_(new_model, factory)
            new_model.load_state_dict(loaded_state_dict, assign=True)

        # Compare the original and loaded models
        for layer_name in ["linear1", "linear2"]:
            original_layer = getattr(model, layer_name)
            new_layer = getattr(new_model, layer_name)

            # Compare weights
            if mode == "weight-only":
                original_weight = original_layer.weight.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = new_layer.weight.tensor_impl.float8_data.to(torch.float32)
            else:
                original_weight = original_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                    torch.float32
                )
                new_weight = (
                    new_layer.weight.original_weight_tensor.tensor_impl.float8_data.to(
                        torch.float32
                    )
                )

            assert torch.allclose(original_weight, new_weight), (
                f"Weights do not match for {layer_name}"
            )

            # Compare scales
            if hasattr(original_layer.weight, "scale"):
                assert torch.allclose(
                    original_layer.weight.scale, new_layer.weight.scale
                ), f"Scales do not match for {layer_name}"

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(
        not is_sm_at_least_89(), "Requires GPU with compute capability >= 8.9"
    )
    def test_fp8_weight_dimension_warning(self):
        # Create model with incompatible dimensions (not multiples of 16)
        model = ToyLinearModel(10, 25).cuda()  # 10x25 and 25x10 weights

        # Set up logging capture
        with self.assertLogs(
            "torchao.quantization.quant_api", level="INFO"
        ) as log_context:
            quantize_(
                model, float8_dynamic_activation_float8_weight(granularity=PerTensor())
            )
            print(model)

        # Verify warning messages for both layers
        expected_messages = [
            "Skipping float8 quantization: weight shape torch.Size([25, 10])",
            "Skipping float8 quantization: weight shape torch.Size([10, 25])",
        ]
        # Check that we got warnings for both incompatible layers
        warning_count = sum(
            1 for msg in log_context.output if "Skipping float8 quantization" in msg
        )
        self.assertEqual(warning_count, 2, "Expected warnings for both linear layers")

        # Check warning message content
        for expected in expected_messages:
            self.assertTrue(
                any(expected in msg for msg in log_context.output),
                f"Expected warning message containing: {expected}",
            )


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])

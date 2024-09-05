from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)
import pytest

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal import common_utils

from torchao.quantization import (
    quantize_,
    float8_weight_only,
    float8_dynamic_activation_float8_weight,
)
from torchao.quantization.observer import PerTensor, PerRow
from torchao.float8.float8_utils import compute_error
import torch
import unittest
import pytest
import copy
import random
from functools import partial
from typing import Tuple
from contextlib import nullcontext
import io


random.seed(0)
torch.manual_seed(0)

is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


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
    @unittest.skipIf(not is_cuda_8_9, "Requires GPU with compute capability >= 8.9")
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.float32])
    @common_utils.parametrize("mode", ["dynamic", "weight-only"])
    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "granularity", [PerTensor(), PerRow()] if is_H100 else [PerTensor()]
    )
    # Inputs are (M,..), K, N
    @common_utils.parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((256,), 512, 256),
            ((64,), 128, 64),
            ((32, 128), 64, 256),
            ((64, 256), 512, 128),
        ],
    )
    def test_fp8_linear_variants(
        self, dtype: torch.dtype, mode: str, compile: bool, sizes: Tuple, granularity
    ):
        raises = (
            isinstance(granularity, PerRow)
            and mode == "dynamic"
            and dtype != torch.bfloat16
        )
        context = (
            nullcontext()
            if not raises
            else pytest.raises(
                AssertionError,
                match="PerRow quantization only works for bfloat16 precision",
            )
        )
        with context:
            M, N, K = sizes
            input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

            mode_map = {
                "dynamic": partial(
                    float8_dynamic_activation_float8_weight, granularity=granularity
                ),
                "weight-only": float8_weight_only,
            }

            # Create a linear layer with bfloat16 dtype
            model = ToyLinearModel(K, N).eval().to(dtype).to("cuda")

            quantized_model = copy.deepcopy(model)
            factory = mode_map[mode]()
            quantize_(model, factory)

            if compile:
                quantized_model = torch.compile(quantized_model, fullgraph=True)

            output_original = model(input_tensor)
            output_quantized = quantized_model(input_tensor)

            error = compute_error(output_original, output_quantized)
            assert (
                compute_error(output_original, output_quantized) > 20
            ), f"Quantization error is too high got a SQNR of {error}"

    def test_invalid_granularity(self):
        with pytest.raises(ValueError, match="Invalid granularity specification"):
            float8_dynamic_activation_float8_weight(granularity="invalid")

    def test_mismatched_granularity(self):
        with pytest.raises(
            ValueError,
            match="Different granularities for activation and weight are not supported",
        ):
            float8_dynamic_activation_float8_weight(granularity=(PerTensor(), PerRow()))

    def test_unsupported_granularity(self):
        class UnsupportedGranularity:
            pass

        with pytest.raises(ValueError, match="Invalid granularity types"):
            float8_dynamic_activation_float8_weight(
                granularity=(UnsupportedGranularity(), UnsupportedGranularity())
            )

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not is_cuda_8_9, "Requires GPU with compute capability >= 8.9")
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
    @unittest.skipIf(not is_cuda_8_9, "Requires GPU with compute capability >= 8.9")
    @common_utils.parametrize("mode", ["dynamic", "weight-only"])
    def test_serialization(self, mode: str):
        # Create and quantize the model
        model = ToyLinearModel(16, 32).to(device="cuda")
        if mode == "dynamic":
            factory = float8_dynamic_activation_float8_weight()
        else:
            factory = float8_weight_only()
        quantize_(model, factory)

        # Save the state dict to an in-memory buffer
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)

        # Reset the buffer position
        buffer.seek(0)

        # Load the state dict from the buffer
        loaded_state_dict = torch.load(buffer)

        # Create a new model and load the state dict
        with torch.device("meta"):
            new_model = ToyLinearModel(16, 32)
            new_model.load_state_dict(loaded_state_dict, assign=True)

        # Compare the original and loaded models
        if mode == "weight-only":
            model_weight_1 = model.linear1.weight.layout_tensor.float8_data.to(
                torch.float32
            )
            new_model_weight_1 = new_model.linear1.weight.layout_tensor.float8_data.to(
                torch.float32
            )

            model_weight_2 = model.linear2.weight.layout_tensor.float8_data.to(
                torch.float32
            )
            new_model_weight_2 = new_model.linear2.weight.layout_tensor.float8_data.to(
                torch.float32
            )

        else:
            model_weight_1 = model.linear1.weight.original_weight_tensor.layout_tensor.float8_data.to(
                torch.float32
            )
            new_model_weight_1 = new_model.linear1.weight.original_weight_tensor.layout_tensor.float8_data.to(
                torch.float32
            )

            model_weight_2 = model.linear2.weight.original_weight_tensor.layout_tensor.float8_data.to(
                torch.float32
            )
            new_model_weight_2 = new_model.linear2.weight.original_weight_tensor.layout_tensor.float8_data.to(
                torch.float32
            )

        assert torch.allclose(model_weight_1, new_model_weight_1)
        assert torch.allclose(model_weight_2, new_model_weight_2)


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])

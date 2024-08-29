from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
    unwrap_tensor_subclass,
)
import pytest

if not TORCH_VERSION_AT_LEAST_2_5:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)

from numpy import full
from torch.testing._internal.common_utils import (
    run_tests,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal import common_utils
from torch._dynamo.testing import CompileCounterWithBackend

from torchao.quantization import (
    quantize_,
    float8_weight_only,
    float8_dynamic_activation_float8_weight,
)
from torchao.float8.float8_utils import compute_error
import torch
import unittest
import pytest
import tempfile
import copy
import random

from unittest.mock import patch


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
        self, dtype: torch.dtype, mode: str, compile: bool, sizes: tuple
    ):
        M, N, K = sizes
        input_tensor = torch.randn(*M, K, dtype=dtype, device="cuda")

        mode_map = {
            "dynamic": float8_dynamic_activation_float8_weight,
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


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])

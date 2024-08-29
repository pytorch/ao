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
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)
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


class TestAffineQuantizedFloat8Basic(TestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_tensor_core_layout_transpose(self):
        l = torch.nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda")
        t = l.weight
        shape = t.shape
        apply_float8_weight_only_quant = float8_weight_only()
        ql = apply_float8_weight_only_quant(l)
        aqt = ql.weight
        aqt_shape = aqt.shape
        assert aqt_shape == shape

        # transpose shape test
        for _ in range(10):
            t = t.t()
            aqt = aqt.t()
            shape = t.shape
            aqt_shape = aqt.shape
            assert aqt_shape == shape

    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    def test_weights_only_save_load(self):
        with torch.no_grad():
            for apply_quant in [float8_weight_only()]:
                # TODO Fails when l requires grad
                l = torch.nn.Linear(128, 256).eval().to(torch.bfloat16).to("cuda")
                ql = apply_quant(l)
                with tempfile.NamedTemporaryFile() as f:
                    torch.save(ql.state_dict(), f)
                    f.seek(0)
                    # `weights_only=True` is enabled for torch 2.5+
                    if TORCH_VERSION_AT_LEAST_2_5:
                        _ = torch.load(f, weights_only=True)
                    else:
                        _ = torch.load(f, weights_only=False)


class TestAffineQuantizedFloat8Compile(InductorTestCase):
    @unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
    @unittest.skipIf(not is_cuda_8_9, "Need H100")
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
    def test_dynamic_fp8_linear(
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

        assert compute_error(output_original, output_quantized) > 20, "Error is too low"


common_utils.instantiate_parametrized_tests(TestAffineQuantizedFloat8Compile)

if __name__ == "__main__":
    pytest.main([__file__])

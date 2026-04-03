# This test file is split from test_float8_mi300.py to allow for separate BUCK
# config for tests that run on AMD and tests that run on Nvidia.

import copy
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import run_tests

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    Granularity,
    PerRow,
    PerTensor,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import get_current_accelerator_device, is_fbcode, is_MI300


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=bias)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    def check_weight_scaling(self, granularity: Granularity):
        qs1 = self.linear1.weight.scale
        qs2 = self.linear2.weight.scale
        N, K = (self.out_features, self.in_features)
        if granularity == PerTensor():
            assert qs1.shape == (1, 1)
            assert qs2.shape == (1, 1)
        elif granularity == PerRow():
            assert qs1.shape == (N, 1)
            assert qs2.shape == (K, 1)


@unittest.skipIf(not is_fbcode(), "fbcode only")
@unittest.skipIf(
    not torch.accelerator.is_available(), "skipping when gpu is not available"
)
@unittest.skipIf(not is_MI300(), "MI300 only")
class TestFloat8MI300(common_utils.TestCase):
    """Dedicated FP8 tests for AMD MI300 hardware."""

    @common_utils.parametrize("compile", [True, False])
    @common_utils.parametrize(
        "granularity",
        [PerTensor(), PerRow()],
    )
    @torch.no_grad()
    def test_fp8_dynamic_matmul(
        self,
        compile: bool,
        granularity: Granularity,
    ):
        dtype = torch.bfloat16
        M, N, K = (128,), 256, 64
        device = get_current_accelerator_device()

        model = ToyLinearModel(K, N, bias=False).eval().to(dtype).to(device)
        quantized_model = copy.deepcopy(model)

        config = Float8DynamicActivationFloat8WeightConfig(
            granularity=granularity,
        )
        quantize_(quantized_model, config)

        quantized_model.check_weight_scaling(granularity)

        if compile:
            quantized_model = torch.compile(quantized_model, fullgraph=True)

        input_tensor = torch.randn(*M, K, dtype=dtype, device=device)
        output_original = model(input_tensor)
        output_quantized = quantized_model(input_tensor)

        error = compute_error(output_original, output_quantized)
        assert error > 20, (
            f"Quantization error: got a SQNR of {error}"
        )


common_utils.instantiate_parametrized_tests(TestFloat8MI300)


if __name__ == "__main__":
    run_tests()

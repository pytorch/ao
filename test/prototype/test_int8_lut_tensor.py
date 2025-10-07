# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import pytest
import torch

from torchao.prototype.parq.quant import (
    StretchedIntxWeightConfig,
    StretchedUnifTorchaoQuantizer,
)
from torchao.prototype.quantization.int8_lut_tensor.int8_lut_tensor import (
    _is_kernel_library_loaded,
)
from torchao.prototype.tensor_conversion.api import _convert_model_for_aarch64
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.utils import compute_error


class ToyLinearModel(torch.nn.Module):
    def __init__(self, d1=512, d2=256, d3=128, d4=8):
        super().__init__()
        self.linear1 = torch.nn.Linear(d1, d2, bias=False)
        self.linear2 = torch.nn.Linear(d2, d3, bias=True)
        self.linear3 = torch.nn.Linear(d3, d4, bias=False)

    def example_inputs(
        self,
        lead_dim=(1,),
        dtype=torch.bfloat16,
    ):
        return torch.randn(
            *lead_dim, self.linear1.in_features, dtype=dtype, device="cpu"
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


@pytest.fixture(autouse=True)
def run_before_and_after_tests():
    yield
    torch._dynamo.reset()  # reset cache between tests


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("granularity", [PerGroup(32), PerAxis(0)])
@pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
@pytest.mark.parametrize("lead_dim", [(5,), (2, 3)])
@pytest.mark.skipif(
    not _is_kernel_library_loaded(), reason="Kernel library is not loaded"
)
def test_parq_conversion(dtype, granularity, bit_width, lead_dim):
    torch.manual_seed(0)
    quantizer = StretchedUnifTorchaoQuantizer(bit_width)
    config = StretchedIntxWeightConfig(
        b=bit_width,
        quant_min=quantizer.quant_min,
        quant_max=quantizer.quant_max,
        granularity=granularity,
        activation_quantization="int8_asym_per_token",
    )

    parq_model = ToyLinearModel(128, 256, 128, 1).to(dtype)
    activations = parq_model.example_inputs(lead_dim=lead_dim, dtype=dtype)
    quantize_(parq_model, config)

    # Convert PARQ model to lowbit LUT model
    lut_model = deepcopy(parq_model)
    _convert_model_for_aarch64(lut_model, tensor_type="int8_lut_tensor")

    # Run both models and compare
    parq_out = parq_model(activations)
    lut_out = lut_model(activations)

    sqnr = compute_error(parq_out, lut_out).item()
    if dtype == torch.float32:
        assert sqnr > 40.0, f"sqnr {sqnr} is too low"
    elif dtype == torch.bfloat16:
        assert sqnr > 25.0, f"sqnr {sqnr} is too low"
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("granularity", [PerGroup(32), PerAxis(0)])
@pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
@pytest.mark.parametrize("lead_dim", [(5,), (2, 3)])
@pytest.mark.skipif(
    not _is_kernel_library_loaded(), reason="Kernel library is not loaded"
)
def test_export(dtype, granularity, bit_width, lead_dim):
    quantizer = StretchedUnifTorchaoQuantizer(bit_width)
    config = StretchedIntxWeightConfig(
        b=bit_width,
        quant_min=quantizer.quant_min,
        quant_max=quantizer.quant_max,
        granularity=granularity,
        activation_quantization="int8_asym_per_token",
    )

    parq_model = ToyLinearModel(128, 256, 128, 8).to(dtype)
    activations = parq_model.example_inputs(lead_dim=lead_dim)
    quantize_(parq_model, config)

    _convert_model_for_aarch64(parq_model)

    ep = torch.export.export(parq_model, (activations,))

    assert (
        f"torch.ops.torchao._linear_8bit_act_{bit_width}bit_weight.default"
        in ep.graph_module.code
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import platform
import sys
from copy import deepcopy
from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from torchao.core.config import AOBaseConfig
from torchao.prototype.parq.quant import StretchedUnifTorchaoQuantizer
from torchao.prototype.parq.quant.quant_api import StretchedIntxWeightOnlyConfig
from torchao.prototype.quantization.dynamic_activation_lut import (
    StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig,
)
from torchao.quantization import quantize_
from torchao.quantization.granularity import PerAxis, PerGroup
from torchao.quantization.linear_activation_quantized_tensor import (
    to_linear_activation_quantized,
)
from torchao.quantization.quant_api import (
    _int8_asymm_per_token_quant,
)
from torchao.quantization.transform_module import register_quantize_module_handler

is_arm64_mac = sys.platform == "darwin" and platform.machine() == "arm64"


@dataclass
class Int8DynamicActivationConfig(AOBaseConfig):
    pass


@register_quantize_module_handler(Int8DynamicActivationConfig)
def _int8_dynamic_activation_transform(
    module: nn.Module, config: Int8DynamicActivationConfig
) -> nn.Module:
    weight = module.weight
    weight = to_linear_activation_quantized(weight, _int8_asymm_per_token_quant)
    module.weight = torch.nn.Parameter(weight, requires_grad=False)
    return module


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
@pytest.mark.skipif(not is_arm64_mac, reason="requires arm64 mac")
def test_parq_conversion(dtype, granularity, bit_width, lead_dim):
    quantizer = StretchedUnifTorchaoQuantizer(bit_width)
    config = StretchedIntxWeightOnlyConfig(
        b=bit_width,
        quant_min=quantizer.quant_min,
        quant_max=quantizer.quant_max,
        granularity=granularity,
    )

    parq_model = ToyLinearModel(128, 256, 128, 1).to(dtype)
    activations = parq_model.example_inputs(lead_dim=lead_dim, dtype=dtype)
    quantize_(parq_model, config)

    # Apply dynamic activation to parq model.  This will serve as the LUT reference
    parq_model_with_dyn_quant = deepcopy(parq_model)
    quantize_(
        parq_model_with_dyn_quant,
        Int8DynamicActivationConfig(),
        # We have to explicitly provide filter_fn because the default linear filter
        # excludes modules with AffinQUnatizedTensor weights
        filter_fn=lambda m, fqn: isinstance(m, torch.nn.Linear),
    )

    # Convert PARQ model to lowbit LUT model
    lut_model = deepcopy(parq_model)
    conversion_config = (
        StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig(
            config.b, config.granularity
        )
    )
    quantize_(lut_model, conversion_config, filter_fn=conversion_config.get_filter_fn())

    # Run both models and compare
    parq_out = parq_model(activations)
    parq_with_dyn_quant_out = parq_model_with_dyn_quant(activations)
    lut_out = lut_model(activations)

    assert torch.allclose(parq_out, parq_with_dyn_quant_out, atol=1e-1, rtol=1e-1)
    if dtype == torch.float32:
        assert torch.allclose(lut_out, parq_with_dyn_quant_out, atol=1e-2, rtol=1e-2)
    elif dtype == torch.bfloat16:
        assert torch.allclose(lut_out, parq_with_dyn_quant_out, atol=1e-2, rtol=1e-2)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("granularity", [PerGroup(32), PerAxis(0)])
@pytest.mark.parametrize("bit_width", [1, 2, 3, 4])
@pytest.mark.parametrize("lead_dim", [(5,), (2, 3)])
@pytest.mark.skipif(not is_arm64_mac, reason="requires arm64 mac")
def test_export(dtype, granularity, bit_width, lead_dim):
    quantizer = StretchedUnifTorchaoQuantizer(bit_width)
    config = StretchedIntxWeightOnlyConfig(
        b=bit_width,
        quant_min=quantizer.quant_min,
        quant_max=quantizer.quant_max,
        granularity=granularity,
    )

    parq_model = ToyLinearModel(128, 256, 128, 8).to(dtype)
    activations = parq_model.example_inputs(lead_dim=lead_dim)
    quantize_(parq_model, config)

    conversion_config = (
        StretchedAffineQuantizedTensor_to_Int8DynamicActivationLutTensorConfig(
            config.b, config.granularity
        )
    )
    quantize_(
        parq_model, conversion_config, filter_fn=conversion_config.get_filter_fn()
    )

    ep = torch.export.export(parq_model, (activations,))
    assert (
        f"torch.ops.torchao._linear_8bit_act_{bit_width}bit_weight.default"
        in ep.graph_module.code
    )

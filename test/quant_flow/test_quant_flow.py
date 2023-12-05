# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# mypy: ignore-errors
import unittest
import torch
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.quant_api import apply_dynamic_quant

def dynamic_quant(model, example_inputs):
    m = capture_pre_autograd_graph(model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
    m = prepare_pt2e(m, quantizer)
    m = convert_pt2e(m)
    return m

def _apply_dynamic_quant(model):
    """
    Applies dynamic symmetric per-token activation and per-channel weight
    quantization to all linear layers in the given model using
    module swaps.
    """
    _replace_with_custom_fn_if_matches_filter(
        model,
        lambda linear_mod: dynamic_quant(linear_mod, (torch.randn(1, linear_mod.in_features))),
        lambda mod, fqn: isinstance(mod, torch.nn.Linear),
    )
    return model


class QuantizerClass:
    pass


def capture_and_prepare(model, example_inputs):
    m = capture_pre_autograd_graph(model, example_inputs)
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_dynamic=True))
    m = prepare_pt2e(m, quantizer)
    # TODO: we can run the weight observer in convert_pt2e so that user don't need to run this
    m(*example_inputs)
    return m

class DynamicQuantizer(QuantizerClass):

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_with_custom_fn_if_matches_filter(
            model,
            lambda linear_mod: capture_and_prepare(linear_mod, (torch.randn(1, linear_mod.in_features))),
            lambda mod, fqn: isinstance(mod, torch.nn.Linear),
        )
        return model

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        _replace_with_custom_fn_if_matches_filter(
            model,
            lambda linear_mod: convert_pt2e(linear_mod),
            lambda mod, fqn: isinstance(mod, torch.fx.GraphModule),
        )
        return model

class EagerDynamicQuantizer(QuantizerClass):

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        apply_dynamic_quant(model)
        return model

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(5, 5).to(torch.float)
        self.linear2 = torch.nn.Linear(5, 5).to(torch.float)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

class TestQuantFlow(unittest.TestCase):
    def test_dynamic_quant_gpu_singleline(self):
        m = M().eval()
        m = _apply_dynamic_quant(m)
        example_inputs = (torch.randn(1, 5).to(dtype=torch.float32),)
        quantized = m(*example_inputs)
        # AssertionError: Expecting input to have dtype torch.float32, but got dtype: torch.float64
        # While executing %choose_qparams_tensor_1 : [num_users=2] = call_function[target=torch.ops.quantized_decomposed.choose_qparams.tensor](args = (%arg0_3, -128, 127, 0.000244140625, torch.int8), kwargs = {})
        # m = torch.compile(m, mode="max-autotune")
        # print(example_inputs[0].dtype)
        # compiled = m(*example_inputs)
        # torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    def test_dynamic_quant_gpu_unified_api_unified_impl(self):
        quantizer = DynamicQuantizer()
        m = M().eval()
        m = quantizer.prepare(m)
        m = quantizer.convert(m)
        example_inputs = (torch.randn(1, 5).to(dtype=torch.float32),)
        quantized = m(*example_inputs)
        # AssertionError: Expecting input to have dtype torch.float32, but got dtype: torch.float64
        # While executing %choose_qparams_tensor_1 : [num_users=2] = call_function[target=torch.ops.quantized_decomposed.choose_qparams.tensor](args = (%arg0_3, -128, 127, 0.000244140625, torch.int8), kwargs = {})
        m = torch.compile(m, mode="max-autotune")
        # print(example_inputs[0].dtype)
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    def test_dynamic_quant_gpu_unified_api_eager_mode_impl(self):
        quantizer = EagerDynamicQuantizer()
        m = M().eval()
        m = quantizer.convert(m)
        example_inputs = (torch.randn(1, 5).to(dtype=torch.float32),)
        quantized = m(*example_inputs)
        m = torch.compile(m, mode="max-autotune")
        compiled = m(*example_inputs)
        torch.testing.assert_close(quantized, compiled, atol=0, rtol=0)

    def test_gptq(self):
        # should be similar to EagerDynamicQuantizer
        pass

if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/3637.

Mirrors the example code in docs/source/eager_tutorials/static_quantization.rst
(minus torch.compile/CUDA, so it runs on CPU in CI) end to end. The
tutorial's example previously had two bugs that made it non-functional if a
reader ran it verbatim, and one deeper design mismatch:
  - QuantizedLinear.from_observed() passed one more positional argument to
    cls(...) than QuantizedLinear.__init__ accepts (an unused `target_dtype`),
    raising TypeError.
  - calculate_qparams() returns scale/zero_point with the block dims
    squeezed out, but Int8Tensor.from_hp requires them to have the same
    number of dims as the tensor being quantized, raising AssertionError.
  - QuantizedLinear.forward() manually pre-quantized the activation into a
    second Int8Tensor and passed *both* quantized tensors into F.linear.
    Int8Tensor's F.linear dispatch has no support for a pre-quantized
    activation tensor (only a plain high-precision one, optionally paired
    with `act_quant_kwargs`/`act_quant_scale`/`act_quant_zero_point` on the
    *weight* tensor for static quantization) - this raised
    `NotImplementedError: ... aten.view ...` from deep inside the dispatch.
    The fix uses the weight tensor's act_quant_kwargs/act_quant_scale/
    act_quant_zero_point fields, which is the currently supported way to do
    static (fixed-scale) activation quantization with Int8Tensor.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.core.config import AOBaseConfig
from torchao.quantization import Int8Tensor, PerRow, PerTensor, quantize_
from torchao.quantization.granularity import PerAxis
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_api import _replace_with_custom_fn_if_matches_filter
from torchao.quantization.quant_primitives import MappingType
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    QuantizeTensorToInt8Kwargs,
)
from torchao.quantization.transform_module import register_quantize_module_handler


class _ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, k, bias=False)
        self.linear2 = torch.nn.Linear(k, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class _ObservedLinear(torch.nn.Linear):
    def __init__(
        self, in_features, out_features, act_obs, weight_obs, bias=True, **kwargs
    ):
        super().__init__(in_features, out_features, bias, **kwargs)
        self.act_obs = act_obs
        self.weight_obs = weight_obs

    def forward(self, input):
        observed_input = self.act_obs(input)
        observed_weight = self.weight_obs(self.weight)
        return F.linear(observed_input, observed_weight, self.bias)

    @classmethod
    def from_float(cls, float_linear, act_obs, weight_obs):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            weight_obs,
            False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear


class _QuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, act_obs, weight_obs, weight, bias):
        super().__init__()
        act_scale, act_zero_point = act_obs.calculate_qparams()
        weight_scale, weight_zero_point = weight_obs.calculate_qparams()
        act_scale = act_scale.reshape(1, 1)
        act_zero_point = act_zero_point.reshape(1, 1)
        weight_scale = weight_scale.reshape(-1, 1)
        weight_zero_point = weight_zero_point.reshape(-1, 1)
        self.bias = bias
        self.qweight = Int8Tensor.from_hp(
            weight,
            granularity=PerRow(),
            scale=weight_scale,
            zero_point=weight_zero_point,
            act_quant_kwargs=QuantizeTensorToInt8Kwargs(granularity=PerTensor()),
            act_quant_scale=act_scale,
            act_quant_zero_point=act_zero_point,
        )

    def forward(self, input):
        return F.linear(input, self.qweight, self.bias)

    @classmethod
    def from_observed(cls, observed_linear):
        return cls(
            observed_linear.in_features,
            observed_linear.out_features,
            observed_linear.act_obs,
            observed_linear.weight_obs,
            observed_linear.weight,
            observed_linear.bias,
        )


@dataclass
class _StaticQuantConfig(AOBaseConfig):
    target_dtype: torch.dtype


@register_quantize_module_handler(_StaticQuantConfig)
def _apply_static_quant(module, config):
    return _QuantizedLinear.from_observed(module)


class TestStaticQuantizationDocExample(TestCase):
    def test_static_quantization_tutorial_example_runs(self):
        dtype = torch.bfloat16
        m = _ToyLinearModel().eval().to(dtype)

        act_obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )
        weight_obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float32,
            zero_point_dtype=torch.float32,
        )

        def insert_observers_(model, act_obs, weight_obs):
            _is_linear = lambda mod, fqn: isinstance(mod, torch.nn.Linear)

            def replacement_fn(mod):
                return _ObservedLinear.from_float(
                    mod, copy.deepcopy(act_obs), copy.deepcopy(weight_obs)
                )

            _replace_with_custom_fn_if_matches_filter(
                model, replacement_fn, _is_linear
            )

        insert_observers_(m, act_obs, weight_obs)

        for _ in range(10):
            example_inputs = m.example_inputs(dtype=dtype)
            m(*example_inputs)

        is_observed_linear = lambda mod, fqn: isinstance(mod, _ObservedLinear)
        # Used to raise TypeError (extra `target_dtype` arg), then
        # AssertionError (scale.ndim mismatch), then
        # NotImplementedError (unsupported pre-quantized activation path)
        # before the doc fix.
        quantize_(m, _StaticQuantConfig(torch.uint8), is_observed_linear)

        self.assertIsInstance(m.linear1, _QuantizedLinear)
        self.assertIsInstance(m.linear2, _QuantizedLinear)
        self.assertIsInstance(m.linear1.qweight, Int8Tensor)
        self.assertEqual(m.linear1.qweight.act_quant_scale.shape, (1, 1))

        # `m.example_inputs()` relies on `self.linear1.in_features`, which no
        # longer exists once `linear1` has been swapped to `_QuantizedLinear`
        # above - construct the input directly instead.
        out = m(torch.randn(1, 64, dtype=dtype))
        self.assertEqual(out.shape, (1, 32))
        self.assertFalse(out.isnan().any())


if __name__ == "__main__":
    run_tests()

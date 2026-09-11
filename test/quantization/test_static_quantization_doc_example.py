# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""Regression test for https://github.com/pytorch/ao/issues/3637.

Mirrors the example code in docs/source/eager_tutorials/static_quantization.rst
(minus torch.compile/CUDA, so it runs on CPU in CI) end to end. The
tutorial's example previously had two bugs that made it non-functional if a
reader ran it verbatim, and two deeper design mismatches:
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
  - QuantizedLinear.__init__ built the activation's QuantizeTensorToInt8Kwargs
    without `mapping_type`, which defaults to MappingType.SYMMETRIC - even
    though `act_obs` above is explicitly MappingType.ASYMMETRIC. Int8Tensor's
    F.linear dispatch only applies the asymmetric zero-point correction when
    `act_quant_kwargs.mapping_type == MappingType.ASYMMETRIC`, so this silently
    produced a finite, correctly-shaped, but numerically wrong result (no
    exception raised) instead of an asymmetrically-quantized one. Fixed by
    passing mapping_type=MappingType.ASYMMETRIC to match `act_obs`, verified
    below by independently recomputing the output from Int8Tensor's own
    documented zero-point-correction formula.
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
from torchao.quantization.quantize_.common.quantize_tensor_kwargs import (
    _choose_quant_func_and_quantize_tensor,
)
from torchao.quantization.quantize_.workflows.int8.int8_tensor import (
    QuantizeTensorToInt8Kwargs,
)
from torchao.quantization.transform_module import register_quantize_module_handler
from torchao.quantization.utils import compute_error


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
            act_quant_kwargs=QuantizeTensorToInt8Kwargs(
                granularity=PerTensor(), mapping_type=MappingType.ASYMMETRIC
            ),
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
        # The doc's act_obs is MappingType.ASYMMETRIC - the tutorial's
        # act_quant_kwargs must match it, or F.linear's asymmetric zero-point
        # correction silently never fires (see module docstring).
        self.assertEqual(
            m.linear1.qweight.act_quant_kwargs.mapping_type, MappingType.ASYMMETRIC
        )
        self.assertEqual(
            m.linear2.qweight.act_quant_kwargs.mapping_type, MappingType.ASYMMETRIC
        )

        # `m.example_inputs()` relies on `self.linear1.in_features`, which no
        # longer exists once `linear1` has been swapped to `_QuantizedLinear`
        # above - construct the input directly instead.
        test_input = torch.randn(1, 64, dtype=dtype)
        out = m(test_input)
        self.assertEqual(out.shape, (1, 32))
        self.assertFalse(out.isnan().any())

        # Numerical check: an attribute-only assertion can't tell a correctly
        # *behaving* asymmetric path from one that merely carries the right
        # enum value but is otherwise miscomputed. Independently recompute
        # linear1's output from Int8Tensor's own documented formula (see the
        # "Asymmetric activation zero_point correction" comment in
        # int8_tensor.py's F.linear dispatch):
        #   Y = (X_int @ W_int^T) * s_x * s_w - zp_x * s_x * row_sum(W_int)^T * s_w
        # using only tensors the fixed code path already produced (qdata/
        # scale on m.linear1.qweight, and the activation quantized via the
        # same _choose_quant_func_and_quantize_tensor helper the dispatch
        # itself calls) - not a second, differently-configured quantization
        # path, which would risk comparing apples to oranges on the (also
        # asymmetric) weight side.
        qweight = m.linear1.qweight
        quantized_input = _choose_quant_func_and_quantize_tensor(
            test_input,
            qweight.act_quant_kwargs,
            scale=qweight.act_quant_scale,
            zero_point=qweight.act_quant_zero_point,
        )
        x_int = quantized_input.qdata.to(torch.float32)
        zp_x = quantized_input.zero_point.reshape(-1, 1).to(torch.float32)
        x_scale = quantized_input.scale.reshape(-1, 1).to(torch.float32)
        w_int = qweight.qdata.to(torch.float32)
        w_scale = qweight.scale.reshape(-1).to(torch.float32)

        y_dot_scaled = (x_int @ w_int.t()) * x_scale
        # With the fix (mapping_type=ASYMMETRIC), the dispatch subtracts this
        # zero-point correction term; without it (the bug this test guards
        # against), the term is silently skipped instead.
        zp_correction = (zp_x * x_scale) * w_int.sum(dim=-1)
        y_corrected = (y_dot_scaled - zp_correction) * w_scale
        y_uncorrected = y_dot_scaled * w_scale

        actual_linear1_out = F.linear(test_input, qweight, None)
        # bfloat16's ~3 decimal digits of precision make elementwise
        # atol/rtol comparisons noisy for a 64-term dot product accumulated
        # in a different order (int8 kernel) vs float32 (this recompute); use
        # the repo's usual SQNR-based comparison instead (matches the >20
        # threshold convention in test_int8_tensor.py/test_integration.py).
        sqnr_vs_corrected = compute_error(actual_linear1_out, y_corrected.to(dtype))
        sqnr_vs_uncorrected = compute_error(actual_linear1_out, y_uncorrected.to(dtype))
        self.assertGreater(
            sqnr_vs_corrected,
            20,
            f"actual output doesn't match the documented zero-point-correction "
            f"formula (sqnr={sqnr_vs_corrected})",
        )
        # This is the check that would have caught the bug: if the
        # correction were silently skipped (mapping_type left at the
        # SYMMETRIC default), actual output would instead match the
        # *uncorrected* formula about as well as it matches the corrected
        # one. Requiring the corrected match to be substantially better
        # rules that out.
        self.assertGreater(
            sqnr_vs_corrected,
            sqnr_vs_uncorrected + 5,
            f"actual output matches the uncorrected (symmetric) formula "
            f"about as well as the corrected one (corrected sqnr="
            f"{sqnr_vs_corrected}, uncorrected sqnr={sqnr_vs_uncorrected}) - "
            f"the asymmetric zero-point correction doesn't appear to be "
            f"having any effect",
        )


if __name__ == "__main__":
    run_tests()

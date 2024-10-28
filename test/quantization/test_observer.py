import re
import unittest

import torch
import torch.nn as nn

# NOTE: we can copy paste these here if we decide to deprecate them in torch.ao
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase

from torchao.quantization.granularity import (
    PerAxis,
    PerTensor,
)
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver,
)
from torchao.quantization.quant_api import (
    insert_observers_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
)


class TestQuantFlow(TestCase):
    def _test_obs_helper(self, obs1, obs2):
        example_inputs = [
            torch.randn(10, 2048),
            torch.randn(10, 2048),
            torch.randn(10, 2048),
        ]
        for example_input in example_inputs:
            obs1(example_input)
            obs2(example_input)

        scale1, zero_point1 = obs1.calculate_qparams()
        scale2, zero_point2 = obs2.calculate_qparams()
        self.assertTrue(torch.allclose(scale1, scale2))
        self.assertTrue(torch.allclose(zero_point1, zero_point2))

    def test_min_max_per_tensor_affine(self):
        obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
        )
        ref_obs = MinMaxObserver(dtype=torch.uint8, qscheme=torch.per_tensor_affine)
        self._test_obs_helper(obs, ref_obs)

    def test_min_max_per_channel_affine(self):
        obs = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
        )
        ref_obs = PerChannelMinMaxObserver(
            dtype=torch.uint8, qscheme=torch.per_channel_affine
        )
        self._test_obs_helper(obs, ref_obs)

    def test_block_size_calc_success(self):
        obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=None,
        )
        example_inputs = [
            torch.randn(10, 2048),
            torch.randn(9, 2048),
            torch.randn(7, 2048),
        ]
        for example_input in example_inputs:
            obs(example_input)

        obs.calculate_qparams()

        obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerAxis(1),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=None,
        )
        for example_input in example_inputs:
            obs(example_input)

        obs.calculate_qparams()

    def test_block_size_row_errors(self):
        obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerAxis(0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=None,
        )
        example_inputs = [
            torch.randn(10, 2048),
            torch.randn(9, 2048),
        ]
        expected_error_msg = "Can't update existing min_val - shape mismatch, self.min_val:torch.Size([10]) != min_val:torch.Size([9])"
        escaped_error_msg = re.escape(expected_error_msg)
        with self.assertRaisesRegex(AssertionError, escaped_error_msg):
            for example_input in example_inputs:
                obs(example_input)

        obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerAxis(1),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=None,
        )
        example_inputs = [
            torch.randn(10, 2048),
            torch.randn(9, 2047),
        ]
        expected_error_msg = "Can't update existing min_val - shape mismatch, self.min_val:torch.Size([2048]) != min_val:torch.Size([2047])"
        escaped_error_msg = re.escape(expected_error_msg)
        with self.assertRaisesRegex(AssertionError, escaped_error_msg):
            for example_input in example_inputs:
                obs(example_input)


class TestLinearObserver(TestCase):
    @common_utils.parametrize("observe_weight", [True, False])
    def test_linear_observer_tensor(self, observe_weight: bool):
        # Create a simple linear layer
        in_features, out_features = 10, 5
        linear = nn.Linear(in_features, out_features)

        # Create observers
        input_observer = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            zero_point_domain=None,
        )
        if observe_weight:
            weight_observer = AffineQuantizedMinMaxObserver(
                MappingType.SYMMETRIC,
                torch.float8_e4m3fn,
                granularity=PerTensor(),
                eps=torch.finfo(torch.float32).eps,
                scale_dtype=torch.float,
                zero_point_dtype=torch.int,
                zero_point_domain=None,
            )
        else:
            weight_observer = None

        # Wrap the weight with LinearObserverTensor
        insert_observers_(linear, input_observer, weight_observer)

        # Create some example inputs
        example_inputs = [torch.randn(5, in_features) for _ in range(3)]
        max_val = 42.1234
        min_val = -39.760
        big_tensor = torch.full((6, in_features), max_val)
        small_tensor = torch.full((40, in_features), min_val)
        example_inputs.extend([big_tensor, small_tensor])

        # Run forward passes
        for example_input in example_inputs:
            _ = linear(example_input)

        input_observer = linear.weight.input_observer

        # Check that the observers have recorded statistics
        assert input_observer.min_val == min_val
        assert input_observer.max_val == max_val

        # Calculate qparams and ensure they're not None
        input_scale, input_zero_point = input_observer.calculate_qparams()

        max_fp8 = torch.finfo(torch.float8_e4m3fn).max
        self.assertEqual(
            input_scale.item(),
            max_val / max_fp8,
        )
        self.assertIsNotNone(input_zero_point)

        if observe_weight:
            weight_observer = linear.weight.weight_observer
            weight_scale, weight_zero_point = weight_observer.calculate_qparams()
            torch.testing.assert_close(
                weight_scale,
                torch.max(linear.weight.original_weight_tensor) / max_fp8,
                atol=5e-5,
                rtol=0.0,
            )
            self.assertIsNotNone(weight_zero_point)
        else:
            self.assertIsNone(linear.weight.weight_observer)


common_utils.instantiate_parametrized_tests(TestLinearObserver)

if __name__ == "__main__":
    unittest.main()

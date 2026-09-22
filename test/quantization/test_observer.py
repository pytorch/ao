# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import re
import unittest

import torch

# NOTE: we can copy paste these here if we decide to deprecate them in torch.ao
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase

from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import (
    AffineQuantizedFixedQParamObserver,
    AffineQuantizedMinMaxObserver,
    AffineQuantizedMSEObserver,
)
from torchao.quantization.quant_primitives import MappingType


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
        )
        for example_input in example_inputs:
            obs(example_input)

        scale, _ = obs.calculate_qparams()  # ignore zero_point for symmetric quant

    def test_block_size_row_errors(self):
        obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerAxis(0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
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

    def test_mse_observer(self):
        obs = AffineQuantizedMSEObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            granularity=PerAxis(0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            steps=100,
            run_once=True,
        )
        example_input = torch.randn(10, 2048)
        obs(example_input)

        scale, _ = obs.calculate_qparams()  # ignore zero_point for symmetric quant

        minmax_obs = AffineQuantizedMinMaxObserver(
            MappingType.SYMMETRIC,
            torch.int8,
            granularity=PerAxis(0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
        )
        minmax_obs(example_input)
        min_val, max_val = minmax_obs.min_val, minmax_obs.max_val
        assert torch.all(
            obs.loss_fn(example_input, obs.min_val, obs.max_val)
            <= obs.loss_fn(example_input, min_val, max_val) + 1e6
        )

    def test_fixed_qparams_observer(self):
        obs = AffineQuantizedFixedQParamObserver(
            MappingType.SYMMETRIC,
            torch.float8_e4m3fn,
            granularity=PerAxis(0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
        )
        example_input = torch.randn(10, 2048)
        obs(example_input)
        obs.set_qparams(torch.ones(2048))
        scale, _ = obs.calculate_qparams()  # ignore zero_point for symmetric quant
        self.assertTrue(torch.allclose(scale, torch.ones(2048)))

    def test_keepdim_per_axis(self):
        """Test keepdim option for per-axis quantization."""
        # Test with keepdim=False (default)
        obs_no_keepdim = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            keepdim=False,
        )
        # Test with keepdim=True
        obs_keepdim = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerAxis(axis=0),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            keepdim=True,
        )

        example_input = torch.randn(10, 2048)
        obs_no_keepdim(example_input)
        obs_keepdim(example_input)

        # Check min_val/max_val shapes differ based on keepdim
        # For PerAxis(0) with input [10, 2048], block_size = [1, 2048]
        # reduction is over dim 1 only
        # With keepdim=False: shape is [10]
        # With keepdim=True: shape is [10, 1]
        self.assertEqual(obs_no_keepdim.min_val.shape, torch.Size([10]))
        self.assertEqual(obs_keepdim.min_val.shape, torch.Size([10, 1]))

        # Calculate qparams
        scale_no_keepdim, zp_no_keepdim = obs_no_keepdim.calculate_qparams()
        scale_keepdim, zp_keepdim = obs_keepdim.calculate_qparams()

        # With keepdim=False: scale/zero_point have reduced shape
        self.assertEqual(scale_no_keepdim.shape, torch.Size([10]))
        self.assertEqual(zp_no_keepdim.shape, torch.Size([10]))

        # With keepdim=True: scale/zero_point keep dimensions (same as min_val/max_val)
        self.assertEqual(scale_keepdim.shape, torch.Size([10, 1]))
        self.assertEqual(zp_keepdim.shape, torch.Size([10, 1]))

        # Values should be the same (just different shapes)
        self.assertTrue(torch.allclose(scale_no_keepdim, scale_keepdim.squeeze()))
        self.assertTrue(
            torch.allclose(zp_no_keepdim.float(), zp_keepdim.squeeze().float())
        )

    @common_utils.parametrize("input_shape", [(10, 2048), (4, 16, 256)])
    def test_keepdim_per_tensor(self, input_shape):
        """Test keepdim option for per-tensor quantization with various input shapes."""
        # Test with keepdim=False (default)
        obs_no_keepdim = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            keepdim=False,
        )
        # Test with keepdim=True
        obs_keepdim = AffineQuantizedMinMaxObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerTensor(),
            eps=torch.finfo(torch.float32).eps,
            scale_dtype=torch.float,
            zero_point_dtype=torch.int,
            keepdim=True,
        )

        example_input = torch.randn(*input_shape)
        obs_no_keepdim(example_input)
        obs_keepdim(example_input)

        # Check min_val/max_val shapes differ based on keepdim
        # For PerTensor, block_size equals input_shape, reduction is over all dims
        # With keepdim=False: min_val shape is [] (scalar)
        # With keepdim=True: min_val shape is [1] * len(input_shape)
        self.assertEqual(obs_no_keepdim.min_val.shape, torch.Size([]))
        self.assertEqual(obs_keepdim.min_val.shape, torch.Size([1] * len(input_shape)))

        # Calculate qparams
        scale_no_keepdim, zp_no_keepdim = obs_no_keepdim.calculate_qparams()
        scale_keepdim, zp_keepdim = obs_keepdim.calculate_qparams()

        # With keepdim=False: scale/zero_point are scalar-like
        self.assertEqual(scale_no_keepdim.shape, torch.Size([]))
        self.assertEqual(zp_no_keepdim.shape, torch.Size([]))

        # With keepdim=True: scale/zero_point keep dimensions (same as min_val/max_val)
        self.assertEqual(scale_keepdim.shape, torch.Size([1] * len(input_shape)))
        self.assertEqual(zp_keepdim.shape, torch.Size([1] * len(input_shape)))

        # Values should be the same (just different shapes)
        self.assertTrue(torch.allclose(scale_no_keepdim, scale_keepdim.squeeze()))
        self.assertTrue(
            torch.allclose(zp_no_keepdim.float(), zp_keepdim.squeeze().float())
        )


common_utils.instantiate_parametrized_tests(TestQuantFlow)

if __name__ == "__main__":
    unittest.main()

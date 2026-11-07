# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import re
import unittest

import torch

# NOTE: we can copy paste these here if we decide to deprecate them in torch.ao
from torch.ao.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    PerChannelMinMaxObserver,
)
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import TestCase

from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import (
    AffineQuantizedFixedQParamObserver,
    AffineQuantizedMinMaxObserver,
    AffineQuantizedMSEObserver,
)
from torchao.quantization.pt2e import observer as pt2e_observer
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
        expected_error_msg = "Observer range shape changed between inputs"
        escaped_error_msg = re.escape(expected_error_msg)
        with self.assertRaisesRegex(ValueError, escaped_error_msg):
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
        expected_error_msg = "Observer range shape changed between inputs"
        escaped_error_msg = re.escape(expected_error_msg)
        with self.assertRaisesRegex(ValueError, escaped_error_msg):
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

        obs = AffineQuantizedFixedQParamObserver(
            MappingType.ASYMMETRIC,
            torch.uint8,
            granularity=PerAxis(0),
            scale_dtype=torch.float32,
            zero_point_dtype=torch.int64,
            scale=torch.tensor([0.5, 0.25], dtype=torch.float64),
            zero_point=torch.tensor([0, 1], dtype=torch.int32),
        )
        scale, zero_point = obs.calculate_qparams()
        self.assertEqual(scale, torch.tensor([0.5, 0.25], dtype=torch.float32))
        self.assertEqual(zero_point, torch.tensor([0, 1], dtype=torch.int64))

        obs.set_qparams(
            torch.tensor([0.125, 0.0625], dtype=torch.float64),
            torch.tensor([2, 3], dtype=torch.int32),
        )
        scale, zero_point = obs.calculate_qparams()
        self.assertEqual(scale, torch.tensor([0.125, 0.0625], dtype=torch.float32))
        self.assertEqual(zero_point, torch.tensor([2, 3], dtype=torch.int64))

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

    @common_utils.parametrize(
        "input_shape",
        [(10, 2048), (1, 2048), (4, 16, 256), (2, 1, 256)],
    )
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


class TestObserverStateDict(TestCase):
    def test_load_restores_versioned_observer_state(self):
        expected_epsilon = 2**-32
        source = torch.nn.Module()
        source.add_module(
            "activation_post_process_0",
            pt2e_observer.MinMaxObserver(eps=expected_epsilon),
        )
        observer_state = pt2e_observer.get_observer_state_dict(source)

        restored = torch.nn.Module()
        restored.add_module("activation_post_process_0", pt2e_observer.MinMaxObserver())
        pt2e_observer.load_observer_state_dict(restored, observer_state)

        self.assertEqual(
            expected_epsilon,
            restored.activation_post_process_0.eps.item(),
        )


def _sparse_far_tail_input():
    # A tail of exactly one quantile step (1e-5 of the elements) far from the
    # bulk: clipping it costs far more than the coarser step of the full range.
    g = torch.Generator().manual_seed(0)
    n, n_zero, n_tail = 1_000_000, 640_000, 10
    bulk = torch.empty(n - n_zero - n_tail).exponential_(4.0, generator=g)
    tail = 35.0 + 105.0 * torch.rand(n_tail, generator=g)
    return torch.cat([torch.zeros(n_zero), bulk, tail])


def _int16_observer(**kwargs):
    return pt2e_observer.HistogramObserver(
        dtype=torch.int16,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-32767,
        quant_max=32767,
        **kwargs,
    )


class TestPT2EHistogramObserver(TestCase):
    def _chosen_and_full_quantization_error(self, obs):
        new_min, new_max = obs._non_linear_param_search()
        bin_width = (obs.max_val - obs.min_val) / obs.bins
        start_bin = int(torch.round((new_min - obs.min_val) / bin_width))
        end_bin = int(torch.round((new_max - obs.min_val) / bin_width)) - 1
        chosen = obs._compute_quantization_error(start_bin, end_bin)
        full = obs._compute_quantization_error(0, obs.bins - 1)
        return new_min, new_max, chosen, full

    def test_default_matches_torch_ao_default(self):
        g = torch.Generator().manual_seed(0)
        symmetric = dict(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
        cases = [
            (dict(dtype=torch.quint8), torch.randn(1000, generator=g)),
            (symmetric, torch.randn(256, generator=g).abs()),
            (symmetric, torch.randn(256, generator=torch.Generator().manual_seed(3))),
            (symmetric, _sparse_far_tail_input()),
        ]
        for kwargs, x in cases:
            obs = pt2e_observer.HistogramObserver(**kwargs)
            ref_obs = HistogramObserver(**kwargs)
            obs(x)
            ref_obs(x)
            self.assertEqual(obs.calculate_qparams(), ref_obs.calculate_qparams())

    def test_default_clips_and_consider_full_range_keeps_sparse_far_tail(self):
        x = _sparse_far_tail_input()
        default_obs = _int16_observer()
        obs = _int16_observer(consider_full_range=True)
        default_obs(x)
        obs(x)
        new_min, new_max, chosen, full = self._chosen_and_full_quantization_error(
            default_obs
        )
        self.assertLess(new_max, x.max())
        self.assertGreater(chosen, full)
        new_min, new_max, chosen, full = self._chosen_and_full_quantization_error(obs)
        self.assertEqual(new_min, x.min())
        self.assertEqual(new_max, x.max())
        self.assertLessEqual(chosen, full)

    def test_consider_full_range_still_clips_when_it_lowers_error(self):
        x = torch.randn(1_000_000, generator=torch.Generator().manual_seed(0))
        obs = pt2e_observer.HistogramObserver(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            consider_full_range=True,
        )
        obs(x)
        new_min, new_max, chosen, full = self._chosen_and_full_quantization_error(obs)
        self.assertGreater(new_min, obs.min_val)
        self.assertLess(new_max, obs.max_val)
        self.assertLess(chosen, full)

    def test_consider_full_range_searches_past_first_trim(self):
        # The first trim raises the error above the full range, but deeper trims
        # lower it below.
        x = torch.randn(256, generator=torch.Generator().manual_seed(3))
        obs = pt2e_observer.HistogramObserver(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric,
            consider_full_range=True,
        )
        obs(x)
        new_min, new_max, chosen, full = self._chosen_and_full_quantization_error(obs)
        self.assertGreater(obs._compute_quantization_error(0, obs.bins - 2), full)
        self.assertLess(new_max, obs.max_val)
        self.assertLess(chosen, full)

    def test_consider_full_range_serialization_and_scripting(self):
        obs = pt2e_observer.HistogramObserver.with_args(consider_full_range=True)()
        self.assertTrue(obs.consider_full_range)
        self.assertEqual(
            set(obs.state_dict()), set(pt2e_observer.HistogramObserver().state_dict())
        )

        x = _sparse_far_tail_input()
        obs = _int16_observer(consider_full_range=True)
        default_obs = _int16_observer()
        obs(x)
        default_obs(x)
        state = obs.__dict__.copy()
        del state["consider_full_range"]
        old_pickle_obs = pt2e_observer.HistogramObserver.__new__(
            pt2e_observer.HistogramObserver
        )
        old_pickle_obs.__setstate__(state)
        self.assertFalse(old_pickle_obs.consider_full_range)
        self.assertEqual(
            old_pickle_obs.calculate_qparams(), default_obs.calculate_qparams()
        )

        ranges = []
        for o in (default_obs, obs):
            scripted = torch.jit.script(o)
            ranges.append(scripted._non_linear_param_search())
            self.assertEqual(ranges[-1], o._non_linear_param_search())
        self.assertNotEqual(ranges[0], ranges[1])


common_utils.instantiate_parametrized_tests(TestQuantFlow)

if __name__ == "__main__":
    unittest.main()

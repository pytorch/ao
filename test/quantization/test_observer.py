import torch
from torch.testing._internal.common_utils import TestCase
from torchao.quantization.observer import (
    AffineQuantizedMinMaxObserver,
    PerTensor,
    PerAxis,
)
from torchao.quantization.quant_primitives import (
    MappingType,
)
import unittest
# NOTE: we can copy paste these here if we decide to deprecate them in torch.ao
from torch.ao.quantization.observer import MinMaxObserver, PerChannelMinMaxObserver

class TestQuantFlow(TestCase):
    def _test_obs_helper(self, obs1, obs2):
        example_inputs = [torch.randn(10, 2048), torch.randn(10, 2048), torch.randn(10, 2048)]
        for example_input in example_inputs:
            obs1(example_input)
            obs2(example_input)

        scale1, zero_point1 = obs1.calculate_qparams()
        scale2, zero_point2 = obs2.calculate_qparams()
        self.assertTrue(torch.allclose(scale1, scale2))
        self.assertTrue(torch.allclose(zero_point1, zero_point2))

    def test_min_max_per_tensor_affine(self):
        obs = AffineQuantizedMinMaxObserver(MappingType.ASYMMETRIC, torch.uint8, granularity_type=PerTensor(), eps=torch.finfo(torch.float32).eps, scale_dtype=torch.float, zero_point_dtype=torch.int)
        ref_obs = MinMaxObserver(dtype=torch.uint8, qscheme=torch.per_tensor_affine)
        self._test_obs_helper(obs, ref_obs)

    def test_min_max_per_channel_affine(self):
        obs = AffineQuantizedMinMaxObserver(MappingType.ASYMMETRIC, torch.uint8, granularity_type=PerAxis(axis=0), eps=torch.finfo(torch.float32).eps, scale_dtype=torch.float, zero_point_dtype=torch.int)
        ref_obs = PerChannelMinMaxObserver(dtype=torch.uint8, qscheme=torch.per_channel_affine)
        self._test_obs_helper(obs, ref_obs)


if __name__ == "__main__":
    unittest.main()

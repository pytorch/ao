# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerRow,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)

FBGEMM_CONFIG = Float8DynamicActivationFloat8WeightConfig(
    granularity=PerRow(), kernel="fbgemm"
)
ATEN_CONFIG = Float8DynamicActivationFloat8WeightConfig(
    granularity=PerRow(), kernel="aten"
)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
class TestFbgemmFp8Tensor(TestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @parametrize("config", [FBGEMM_CONFIG, ATEN_CONFIG])
    def test_linear(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, config)
        quantized = linear(input)
        sqnr = compute_error(original, quantized)
        self.assertTrue(sqnr > 20, f"sqnr: {sqnr}")

    @parametrize("config", [FBGEMM_CONFIG, ATEN_CONFIG])
    def test_slice(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)
        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )
        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, config)
        weight1 = dummy.weight.narrow(0, 0, 64)
        weight2 = dummy.weight.narrow(1, 0, 128)
        self.assertEqual(weight1.float8_data, dummy.weight.float8_data.narrow(0, 0, 64))
        self.assertEqual(weight1.scale, dummy.weight.scale.narrow(0, 0, 64))
        self.assertEqual(
            weight2.float8_data, dummy.weight.float8_data.narrow(1, 0, 128)
        )
        self.assertEqual(weight2.scale, dummy.weight.scale)

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 25, f"sqnr: {sqnr}")

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        self.assertTrue(sqnr > 15, f"sqnr: {sqnr}")

    @parametrize("config", [FBGEMM_CONFIG, ATEN_CONFIG])
    def test_slice_and_copy_(self, config):
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        assert param.data.float8_data.data_ptr() == param_data.float8_data.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()
        orig_value = param.data.float8_data[0][0].item()

        # dummy_l has random input (shouldn't be 0)
        dummy_l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        quantize_(dummy_l, config)
        quantized = dummy_l.weight
        quantized = quantized.narrow(0, 0, 512)

        param_data.copy_(quantized)

        # making sure param.data is updated
        assert param.data.float8_data[0][0] != orig_value

    @parametrize("config", [FBGEMM_CONFIG])
    def test_bmm(self, config):
        class M(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = weight

            def forward(self, x):
                return torch.bmm(x, self.weight)

        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(10, 32, 128, dtype=dtype, device=device)
        weight = torch.randn(10, 128, 256, dtype=dtype, device=device)
        m = M(weight).eval()
        original = m(input)
        # we need to transpose the weight first for bmm
        m.weight = torch.nn.Parameter(m.weight.transpose(1, 2).contiguous())
        quantize_(m, config, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    @parametrize("config", [FBGEMM_CONFIG, ATEN_CONFIG])
    def test_to_device(self, config):
        for device in self.GPU_DEVICES:
            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, config)
            linear.to(device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, config)
            linear.to(device=device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, config)
            linear.to(device)

    @parametrize("config", [FBGEMM_CONFIG, ATEN_CONFIG])
    def test_cat(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        # weight: (256, 128)
        linear1 = torch.nn.Linear(128, 256, dtype=dtype)
        # weight: (256, 128)
        linear2 = torch.nn.Linear(128, 256, dtype=dtype)

        cat_weight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        dummy1 = torch.nn.Linear(128, 512, bias=False, dtype=dtype, device=device)

        dummy1.weight = torch.nn.Parameter(cat_weight1)
        quantize_(dummy1, config)

        quantize_(linear1, config)
        quantize_(linear2, config)

        cat_qweight1 = torch.cat([linear1.weight, linear2.weight], dim=0)
        self.assertTrue(cat_qweight1.shape, (512, 128))
        self.assertEqual(dummy1.weight.float8_data, cat_qweight1.float8_data)
        self.assertEqual(dummy1.weight.scale, cat_qweight1.scale)

        # concat with dim == 1 is not really correct and will be fixed later
        # when we support distributed checkpointing
        cat_qweight2 = torch.cat([linear1.weight, linear2.weight], dim=1)
        self.assertTrue(cat_qweight2.shape, (256, 256))
        ref_float8_data = torch.cat(
            [linear1.weight.float8_data, linear2.weight.float8_data], dim=1
        )
        ref_scale = linear1.weight.scale
        self.assertEqual(cat_qweight2.float8_data, ref_float8_data)
        self.assertEqual(cat_qweight2.scale, ref_scale)

    @parametrize("config", [FBGEMM_CONFIG])
    def test_transpose(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        # weight: (256, 128)
        linear1 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear1, config)
        linear1.weight = torch.nn.Parameter(linear1.weight.transpose(0, 1).contiguous())
        linear1.bias = torch.nn.Parameter(torch.randn(128, dtype=dtype, device=device))
        self.assertTrue(linear1.weight.shape, (128, 256))

        input = torch.randn(32, 256, dtype=dtype, device=device)
        # make sure it runs
        res = linear1(input)
        self.assertTrue(res.shape, (32, 128))


instantiate_parametrized_tests(TestFbgemmFp8Tensor)


if __name__ == "__main__":
    run_tests()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao.quantization import (
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
class TestIntxUnpackedTensor(TestCase):
    def setUp(self):
        self.config = IntxWeightOnlyConfig(
            weight_dtype=torch.int4,
            granularity=PerGroup(32),
            version=2,
        )

    def test_embedding(self):
        dtype = torch.bfloat16
        device = "cpu"
        input = torch.randint(low=0, high=128, size=(10,), device=device)
        embedding = torch.nn.Embedding(128, 256, dtype=dtype, device=device)
        original = embedding(input)
        quantize_(embedding, self.config)
        quantized = embedding(input)
        error = compute_error(original, quantized)
        self.assertTrue(error > 20)

    def test_linear(self):
        dtype = torch.bfloat16
        device = "cpu"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, self.config)
        quantized = linear(input)
        error = compute_error(original, quantized)
        self.assertTrue(error > 20)

    def test_slice(self):
        dtype = torch.bfloat16
        device = "cpu"
        dummy = torch.nn.Linear(256, 256, bias=False, dtype=dtype, device=device)

        dummy1 = torch.nn.Linear(256, 64, bias=False, dtype=dtype, device=device)
        dummy1.weight = torch.nn.Parameter(
            dummy.weight.narrow(0, 0, 64), requires_grad=False
        )

        dummy2 = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        dummy2.weight = torch.nn.Parameter(
            dummy.weight.narrow(1, 0, 128), requires_grad=False
        )

        quantize_(dummy, self.config)
        weight1 = dummy.weight.narrow(0, 0, 64)
        weight2 = dummy.weight.narrow(1, 0, 128)

        self.assertEqual(weight1.qdata, dummy.weight.qdata.narrow(0, 0, 64))
        self.assertEqual(weight1.scale, dummy.weight.scale.narrow(0, 0, 64))

        self.assertEqual(weight2.qdata, dummy.weight.qdata.narrow(1, 0, 128))
        self.assertEqual(weight2.scale, dummy.weight.scale.narrow(1, 0, 4))

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1, requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 20

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 15

    def test_slice_and_copy_(self):
        device = "cpu"
        l = torch.nn.Linear(1024, 1024).to(device).to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device=device)
        )
        quantize_(l, self.config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        assert param.data.qdata.data_ptr() == param_data.qdata.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()
        assert param.data.zero_point.data_ptr() == param_data.zero_point.data_ptr()
        orig_value = param.data.qdata[0][0].item()

        # dummy_l has random input (shouldn't be 0)
        dummy_l = torch.nn.Linear(1024, 1024).to(device).to(torch.bfloat16)
        quantize_(dummy_l, self.config)
        quantized = dummy_l.weight
        quantized = quantized.narrow(0, 0, 512)

        param_data.copy_(quantized)

        # making sure param.data is updated
        assert param.data.qdata[0][0] != orig_value

    def test_to_dtype(self):
        activations_bf16 = torch.randn(1, 128, dtype=torch.bfloat16)
        activations_fp32 = torch.randn(1, 128, dtype=torch.float32)
        activations_fp16 = torch.randn(1, 128, dtype=torch.float16)

        linear = torch.nn.Linear(128, 256)
        quantize_(linear, self.config)

        linear.to(dtype=torch.float16)
        linear(activations_fp16)

        linear.to(dtype=torch.float32)
        linear(activations_fp32)

        linear.to(dtype=torch.bfloat16)
        linear(activations_bf16)

    def test_export(self):
        linear = torch.nn.Linear(128, 256)
        quantize_(linear, self.config)
        ep = torch.export.export(linear, (torch.randn(1, 128),))
        assert "torch.ops.torchao.dequantize_affine.default" in ep.graph_module.code


if __name__ == "__main__":
    run_tests()

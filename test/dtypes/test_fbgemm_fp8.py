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

from torchao.float8.config import e4m3_dtype
from torchao.quantization import (
    FbgemmConfig,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
class TestFbgemmFp8Tensor(TestCase):
    def setUp(self):
        self.config = FbgemmConfig(
            input_dtype=e4m3_dtype,
            weight_dtype=e4m3_dtype,
            output_dtype=torch.bfloat16,
        )

    def test_linear(self):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, self.config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    def test_slice(self):
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

        quantize_(dummy, self.config)
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
        assert compute_error(res, res_ref) > 25

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        assert compute_error(res, res_ref) > 15

    def test_slice_and_copy_(self):
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, self.config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        assert param.data.float8_data.data_ptr() == param_data.float8_data.data_ptr()
        assert param.data.scale.data_ptr() == param_data.scale.data_ptr()
        orig_value = param.data.float8_data[0][0].item()

        # dummy_l has random input (shouldn't be 0)
        dummy_l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        quantize_(dummy_l, self.config)
        quantized = dummy_l.weight
        quantized = quantized.narrow(0, 0, 512)

        param_data.copy_(quantized)

        # making sure param.data is updated
        assert param.data.float8_data[0][0] != orig_value


if __name__ == "__main__":
    run_tests()

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
class TestInt4GroupwisePreshuffleTensor(TestCase):
    def setUp(self):
        self.config = FbgemmConfig(
            input_dtype=torch.bfloat16,
            weight_dtype=torch.int4,
            output_dtype=torch.bfloat16,
            block_size=[1, 128],
            preshuffle=True,
        )
        self.bmm_config = FbgemmConfig(
            input_dtype=torch.bfloat16,
            weight_dtype=torch.int4,
            output_dtype=torch.bfloat16,
            block_size=[1, 1, 128],
            preshuffle=True,
        )
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    def test_linear(self):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, self.config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    @unittest.skip("WIP: this doesn't work yet")
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
        self.assertEqual(
            weight1.packed_weight, dummy.weight.packed_weight.narrow(0, 0, 64)
        )
        self.assertEqual(weight1.group_scale, dummy.weight.group_scale.narrow(1, 0, 64))
        self.assertEqual(
            weight2.packed_weight, dummy.weight.packed_weight.narrow(1, 0, 64)
        )
        self.assertEqual(weight2.group_scale, dummy.weight.group_scale.narrow(0, 0, 1))

        # check for sliced weight, before and after float8 quantization
        # does not differ too much
        input = torch.randn(2, 256, dtype=dtype, device=device)
        res_ref = dummy1(input)
        dummy.weight = torch.nn.Parameter(weight1, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        assert sqnr > 20, f"Got: {sqnr}"

        input = torch.randn(2, 128, dtype=dtype, device=device)
        res_ref = dummy2(input)
        dummy.weight = torch.nn.Parameter(weight2, requires_grad=False)
        res = dummy(input)
        sqnr = compute_error(res, res_ref)
        assert sqnr > 15, f"Got: {sqnr}"

    def test_slice_and_copy_(self):
        l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        l.weight = torch.nn.Parameter(
            torch.zeros(1024, 1024, dtype=torch.bfloat16, device="cuda")
        )
        quantize_(l, self.config)
        param = l.weight
        param_data = param.data
        param_data = param_data.narrow(0, 0, 512)
        assert (
            param.data.packed_weight.data_ptr() == param_data.packed_weight.data_ptr()
        )
        assert param.data.group_scale.data_ptr() == param_data.group_scale.data_ptr()
        assert param.data.row_scale.data_ptr() == param_data.row_scale.data_ptr()
        orig_value = param.data.packed_weight[0][0].item()

        # dummy_l has random input (shouldn't be 0)
        dummy_l = torch.nn.Linear(1024, 1024).to("cuda").to(torch.bfloat16)
        quantize_(dummy_l, self.config)
        quantized = dummy_l.weight
        quantized = quantized.narrow(0, 0, 512)

        param_data.copy_(quantized)

        # making sure param.data is updated
        assert param.data.packed_weight[0][0] != orig_value

    def test_bmm(self):
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
        m.weight = torch.nn.Parameter(m.weight.transpose(1, 2).contiguous())
        quantize_(m, self.bmm_config, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 18)

    def test_to_device(self):
        for device in self.GPU_DEVICES:
            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, self.config)
            linear.to(device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, self.config)
            linear.to(device=device)

            linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
            quantize_(linear, self.config)
            linear.to(device)


if __name__ == "__main__":
    run_tests()

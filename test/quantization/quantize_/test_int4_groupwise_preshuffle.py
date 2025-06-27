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
    _is_fbgemm_genai_gpu_available,
    is_sm_at_least_90,
)


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
@unittest.skipIf(
    not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
)
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

    def test_module_path(self):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear, self.config)
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4GroupwisePreshuffleTensor'>",
        )


if __name__ == "__main__":
    run_tests()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import (
    Int4WeightOnlyConfig,
    quantize_,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    torch_version_at_least,
)


def get_config(group_size):
    return Int4WeightOnlyConfig(
        group_size=group_size,
        packing_format="int4_woq_cpu",
        version=2,
    )


@unittest.skipIf(not torch_version_at_least("2.6.0"), "Need pytorch 2.6+")
class TestInt4WoqCpuTensor(TestCase):
    @parametrize("group_size", [32, 64, 128])
    def test_linear(self, group_size):
        dtype = torch.bfloat16
        device = "cpu"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size))
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    @parametrize("group_size", [32, 64, 128])
    def test_module_path(self, group_size):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear, get_config(group_size))
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4WoqCpuTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4WoqCpuTensor'>",
            )


instantiate_parametrized_tests(TestInt4WoqCpuTensor)


if __name__ == "__main__":
    run_tests()

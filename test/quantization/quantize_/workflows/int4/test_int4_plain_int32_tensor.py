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
        packing_format="plain_int32",
        version=2,
    )


@unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch 2.8+")
@unittest.skipIf(not torch.xpu.is_available(), "CUDA not available")
class Int4PlainInt32Tensor(TestCase):
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 12),
        ],
    )
    @parametrize("dtype", [torch.bfloat16, torch.half])
    @parametrize("group_size", [32, 64, 128])
    def test_linear(self, sizes, dtype, group_size):
        device = "xpu"
        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size))
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

        compiled_linear = torch.compile(linear)
        quantized_and_compiled = compiled_linear(input)
        self.assertTrue(compute_error(original, quantized_and_compiled) > 20)

    @parametrize("dtype", [torch.bfloat16, torch.half])
    def test_module_path(self, dtype):
        linear = torch.nn.Linear(128, 256, dtype=dtype, device="xpu")
        quantize_(linear, get_config(group_size=128))
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4PlainInt32Tensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4PlainInt32Tensor'>",
            )


instantiate_parametrized_tests(Int4PlainInt32Tensor)


if __name__ == "__main__":
    run_tests()

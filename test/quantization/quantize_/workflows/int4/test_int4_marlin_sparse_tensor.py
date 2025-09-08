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
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.testing.utils import skip_if_rocm
from torchao.utils import torch_version_at_least

BF16_ACT_CONFIG = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="marlin_sparse",
    version=2,
)


@unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
class TestInt4MarlinSparseTensor(TestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @skip_if_rocm("ROCm enablement in progress")
    @parametrize("config", [BF16_ACT_CONFIG])
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 12),
        ],
    )
    def test_linear(self, config, sizes):
        dtype = torch.float16
        device = "cuda"

        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)

        apply_fake_sparsity(linear)
        original = linear(input)
        quantize_(linear, config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

        compiled_linear = torch.compile(linear)
        quantized_and_compiled = compiled_linear(input)
        self.assertTrue(compute_error(original, quantized_and_compiled) > 20)

    @skip_if_rocm("ROCm enablement in progress")
    @unittest.skip("Fix later")
    @parametrize("config", [BF16_ACT_CONFIG])
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

    @skip_if_rocm("ROCm enablement in progress")
    @parametrize("config", [BF16_ACT_CONFIG])
    def test_module_path(self, config):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear.cuda(), config)
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4MarlinSparseTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4MarlinSparseTensor'>",
            )


instantiate_parametrized_tests(TestInt4MarlinSparseTensor)


if __name__ == "__main__":
    run_tests()

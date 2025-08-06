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

from torchao.float8.config import e4m3_dtype
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

if TORCH_VERSION_AT_LEAST_2_8:
    BF16_ACT_CONFIG = FbgemmConfig(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.int4,
        output_dtype=torch.bfloat16,
        block_size=[1, 128],
        preshuffle=True,
    )

    BF16_ACT_BMM_CONFIG = FbgemmConfig(
        input_dtype=torch.bfloat16,
        weight_dtype=torch.int4,
        output_dtype=torch.bfloat16,
        block_size=[1, 1, 128],
        preshuffle=True,
    )

    FP8_ACT_CONFIG = FbgemmConfig(
        input_dtype=e4m3_dtype,
        weight_dtype=torch.int4,
        output_dtype=torch.bfloat16,
        block_size=[1, 128],
        preshuffle=True,
    )

    FP8_ACT_BMM_CONFIG = FbgemmConfig(
        input_dtype=e4m3_dtype,
        weight_dtype=torch.int4,
        output_dtype=torch.bfloat16,
        block_size=[1, 1, 128],
        preshuffle=True,
    )

else:
    BF16_ACT_CONFIG = None
    BF16_ACT_BMM_CONFIG = None
    FP8_ACT_CONFIG = None
    FP8_ACT_BMM_CONFIG = None


@unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
@unittest.skipIf(not is_sm_at_least_90(), "Nedd sm90+")
@unittest.skipIf(
    not _is_fbgemm_genai_gpu_available(), "Requires fbgemm-gpu-genai >= 1.2.0"
)
class TestInt4PreshuffledTensor(TestCase):
    def setUp(self):
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @parametrize("config", [BF16_ACT_CONFIG, FP8_ACT_CONFIG])
    def test_linear(self, config):
        dtype = torch.bfloat16
        device = "cuda"
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, config)
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

    # Note: this order will error out: `Got bad cuda status: an illegal memory access was encountered at line: 449`
    # @parametrize("bmm_config", [BF16_ACT_BMM_CONFIG, FP8_ACT_BMM_CONFIG])
    @parametrize("bmm_config", [FP8_ACT_BMM_CONFIG, BF16_ACT_BMM_CONFIG])
    def test_bmm(self, bmm_config):
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
        quantize_(m, bmm_config, filter_fn=lambda x, fqn: True)
        quantized = m(input)
        self.assertTrue(compute_error(original, quantized) > 18)

    @parametrize("config", [BF16_ACT_CONFIG, FP8_ACT_CONFIG])
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

    @parametrize("config", [BF16_ACT_CONFIG, FP8_ACT_CONFIG])
    def test_module_path(self, config):
        linear = torch.nn.Linear(128, 256, dtype=torch.bfloat16)
        quantize_(linear, config)
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4PreshuffledTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4PreshuffledTensor'>",
            )


instantiate_parametrized_tests(TestInt4PreshuffledTensor)


if __name__ == "__main__":
    run_tests()

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
from torchao.quantization.quantize_.common import SupportsActivationPreScaling
from torchao.quantization.utils import compute_error
from torchao.utils import (
    torch_version_at_least,
)


def get_config(group_size, use_hqq):
    return Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format="opaque",
        int4_choose_qparams_algorithm="hqq" if use_hqq else "tinygemm",
    )


@unittest.skipIf(not torch_version_at_least("2.6.0"), "Need pytorch 2.6+")
class TestInt4OpaqueTensor(TestCase):
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 12),
        ],
    )
    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @parametrize("group_size", [32, 64, 128])
    @parametrize("use_hqq", [True, False])
    def test_linear(self, sizes, dtype, group_size, use_hqq):
        device = "cpu"
        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size, use_hqq))
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > 20)

        compiled_linear = torch.compile(linear)
        quantized_and_compiled = compiled_linear(input)
        self.assertTrue(compute_error(original, quantized_and_compiled) > 20)

    @parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @parametrize("use_hqq", [True, False])
    def test_module_path(self, dtype, use_hqq):
        linear = torch.nn.Linear(128, 256, dtype=dtype)
        quantize_(linear, get_config(group_size=128, use_hqq=use_hqq))
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4OpaqueTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4OpaqueTensor'>",
            )

    @parametrize("use_hqq", [True, False])
    def test_activation_prescaling(self, use_hqq):
        dtype = torch.bfloat16
        input = torch.randn(1, 128, dtype=dtype)
        linear = torch.nn.Linear(128, 256, bias=False, dtype=dtype)
        original_output = linear(input)
        quantize_(linear, get_config(group_size=128, use_hqq=use_hqq))
        qw = linear.weight
        assert isinstance(qw, SupportsActivationPreScaling), (
            "Expected int4 tensor supports activation prescaling"
        )
        assert qw.act_pre_scale is None, "Default `act_pre_scale` is None"
        _ACT_PRE_SCALE = 2
        manual_scaled_quantized = linear(input * _ACT_PRE_SCALE)
        qw.act_pre_scale = _ACT_PRE_SCALE
        auto_scaled_quantized = linear(input)

        # Making sure activation pre scaling is successfully applied to the activation.
        self.assertEqual(manual_scaled_quantized, auto_scaled_quantized)

        # If pre-scaling is auto-applied, the quantization error should be low,
        # i.e., compute_error (SQNR) is high
        self.assertTrue(
            compute_error(original_output * _ACT_PRE_SCALE, auto_scaled_quantized) > 20
        )


instantiate_parametrized_tests(TestInt4OpaqueTensor)


if __name__ == "__main__":
    run_tests()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import tempfile
from packaging import version

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

try:
    import torch_npu
except ImportError:
    torch_npu = None


def get_config(group_size):
    return Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format="plain_int32",
    )


@unittest.skipIf(not torch_version_at_least("2.7.1"), "Need pytorch 2.7.1+")
@unittest.skipIf(torch_npu is None, "torch_npu is not available")
@unittest.skipIf(not torch_npu.npu.is_available(), "NPU not available")
@unittest.skipIf(
    version.parse(torch_npu.__version__) < version.parse("2.7.1rc1"),
    "Need torch_npu 2.7.1rc1+",
)
class Int4PlainInt32TensorNPU(TestCase):

    @parametrize("device", ["npu"])
    @parametrize(
        "sizes",
        [
            ((128,), 256, 128),
            ((32, 128), 512, 128),
            ((2, 32, 128), 256, 128),
        ],
    )
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("group_size", [32, 64])
    def test_linear(self, device, sizes, dtype, group_size):
        M, N, K = sizes
        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        orig_output = linear(input)
        quantize_(linear, get_config(group_size))
        quantized_output = linear(input)
        self.assertTrue(compute_error(orig_output, quantized_output) > 10)

    @parametrize("device", ["npu"])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_module_path(self, device, dtype):
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, get_config(group_size=64))
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Int4PlainInt32TensorNPU'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Int4PlainInt32TensorNPU'>",
            )

    @parametrize("device", ["npu"])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_activation_prescaling(self, device, dtype):
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, bias=False, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(64))
        qw = linear.weight
        assert isinstance(
            qw, SupportsActivationPreScaling
        ), "Expected int4 tensor supports activation prescaling"
        assert qw.act_pre_scale is None, "Default `act_pre_scale` is None"
        _ACT_PRE_SCALE = 2
        qw.act_pre_scale = _ACT_PRE_SCALE
        quantized = linear(input)

        # making sure activation pre scaling is successfully applied to the activation
        self.assertTrue(compute_error(original * _ACT_PRE_SCALE, quantized) > 10)


instantiate_parametrized_tests(Int4PlainInt32TensorNPU)

if __name__ == "__main__":
    run_tests()

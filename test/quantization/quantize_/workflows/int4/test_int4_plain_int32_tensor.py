# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import pytest
import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    TestCase,
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


def get_config(group_size, use_hqq=False):
    return Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format="plain_int32",
        int4_choose_qparams_algorithm="hqq" if use_hqq else "tinygemm",
    )


class Int4PlainInt32Tensor(TestCase):
    _MIN_VER = {
        "xpu": "2.8.0",
        "npu": "2.7.1",
    }

    def setUp(self):
        min_req = type(self)._MIN_VER.get(self.device_type)
        if not torch_version_at_least(min_req):
            self.skipTest(
                f"{self.device_type} requires torch >= {min_req}, current {torch.__version__}"
            )

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
    @parametrize("use_hqq", [False, True])
    @parametrize("thresholds", [{"xpu": 20, "npu": 10}])
    def test_linear(self, device, sizes, dtype, group_size, use_hqq, thresholds):
        M, N, K = sizes
        if "npu" in device and group_size == K:
            pytest.skip(
                f"{device} does not support group_size equal to K dimension ({group_size} == {K})"
            )
        threshold = thresholds.get(device.split(":")[0])

        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size, use_hqq))
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > threshold)

        if "xpu" in device:
            compiled_linear = torch.compile(linear)
            quantized_and_compiled = compiled_linear(input)
            self.assertTrue(compute_error(original, quantized_and_compiled) > threshold)

    @parametrize("dtype", [torch.bfloat16, torch.half])
    @parametrize("use_hqq", [False, True])
    def test_module_path(self, device, dtype, use_hqq):
        K, N, group_size = 128, 256, 128
        if "npu" in device:
            group_size = 64

        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        quantize_(linear, get_config(group_size, use_hqq))
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

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    @parametrize("use_hqq", [False, True])
    @parametrize("thresholds", [{"xpu": 20, "npu": 10}])
    def test_activation_prescaling(self, device, dtype, use_hqq, thresholds):
        if "xpu" in device and dtype == torch.float16:
            pytest.skip(f"{device} test_activation_prescaling don't test {dtype}")

        threshold = thresholds.get(device.split(":")[0])
        K, N, group_size = 128, 256, 128
        if "npu" in device:
            group_size = 64

        input = torch.randn(1, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size, use_hqq))
        qw = linear.weight
        assert isinstance(qw, SupportsActivationPreScaling), (
            "Expected int4 tensor supports activation prescaling"
        )
        assert qw.act_pre_scale is None, "Default `act_pre_scale` is None"
        _ACT_PRE_SCALE = 2
        qw.act_pre_scale = _ACT_PRE_SCALE
        quantized = linear(input)

        # making sure activation pre scaling is successfully applied to the activation
        self.assertTrue(compute_error(original * _ACT_PRE_SCALE, quantized) > threshold)


instantiate_device_type_tests(
    Int4PlainInt32Tensor, globals(), only_for=("xpu", "npu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()

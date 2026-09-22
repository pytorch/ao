# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
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


def get_config(group_size):
    return Int4WeightOnlyConfig(
        group_size=group_size,
        int4_packing_format="plain_int32",
    )


class GroupedMMModel(torch.nn.Module):
    """A toy model whose only op in forward is torch._grouped_mm."""

    def __init__(self, E, K, N, device, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(E, N, K, device=device, dtype=dtype)
        )

    def forward(self, x, offs):
        return torch._grouped_mm(x, self.weight.transpose(-2, -1), offs=offs)


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
    @parametrize("thresholds", [{"xpu": 20, "npu": 10}])
    def test_linear(self, device, sizes, dtype, group_size, thresholds):
        M, N, K = sizes
        if "npu" in device and group_size == K:
            pytest.skip(
                f"{device} does not support group_size equal to K dimension ({group_size} == {K})"
            )
        threshold = thresholds.get(device.split(":")[0])

        input = torch.randn(*M, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size))
        quantized = linear(input)
        self.assertTrue(compute_error(original, quantized) > threshold)

        if "xpu" in device:
            compiled_linear = torch.compile(linear)
            quantized_and_compiled = compiled_linear(input)
            self.assertTrue(compute_error(original, quantized_and_compiled) > threshold)

    @parametrize("dtype", [torch.bfloat16, torch.half])
    def test_module_path(self, device, dtype):
        K, N, group_size = 128, 256, 128
        if "npu" in device:
            group_size = 64

        linear = torch.nn.Linear(K, N, dtype=dtype, device=device)
        quantize_(linear, get_config(group_size))
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
    @parametrize("thresholds", [{"xpu": 20, "npu": 10}])
    def test_activation_prescaling(self, device, dtype, thresholds):
        if "xpu" in device and dtype == torch.float16:
            pytest.skip(f"{device} test_activation_prescaling don't test {dtype}")

        threshold = thresholds.get(device.split(":")[0])
        K, N, group_size = 128, 256, 128
        if "npu" in device:
            group_size = 64

        input = torch.randn(1, K, dtype=dtype, device=device)
        linear = torch.nn.Linear(K, N, bias=False, dtype=dtype, device=device)
        original = linear(input)
        quantize_(linear, get_config(group_size))
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

    @parametrize("dtype", [torch.bfloat16, torch.half])
    @parametrize("group_size", [32, 64, 128])
    def test_grouped_mm(self, device, dtype, group_size):
        """Test Int4WeightOnlyConfig (plain_int32) with grouped_mm dispatch
        (weight-only dequant path, since there is no native int4 grouped_mm kernel).
        """
        # local import to avoid collision with the `Int4PlainInt32Tensor`
        # TestCase class defined in this module
        from torchao.quantization import Int4PlainInt32Tensor

        if "npu" in device:
            pytest.skip("grouped_mm is only supported on XPU for Int4PlainInt32Tensor")

        E, K, N = 4, 256, 256
        m_per_group = [32, 64, 16, 48]
        total_m = sum(m_per_group)

        model_ref = GroupedMMModel(E, K, N, device=device, dtype=dtype)
        model = copy.deepcopy(model_ref)

        x = torch.randn(total_m, K, device=device, dtype=dtype)
        offs = torch.tensor(
            [sum(m_per_group[: i + 1]) for i in range(E)],
            device=device,
            dtype=torch.int32,
        )

        y_ref = model_ref(x, offs)

        quantize_(
            model,
            get_config(group_size),
            filter_fn=lambda mod, fqn: (
                isinstance(mod, GroupedMMModel) and hasattr(mod, "weight")
            ),
        )

        self.assertIsInstance(model.weight, Int4PlainInt32Tensor)

        w_sqnr = compute_error(model_ref.weight, model.weight.dequantize())
        self.assertGreater(w_sqnr, 19.0, f"Weight SQNR too low: {w_sqnr:.2f}")

        y = model(x, offs)
        y_sqnr = compute_error(y_ref, y)
        self.assertGreater(y_sqnr, 15.0, f"Output SQNR too low: {y_sqnr:.2f}")


instantiate_device_type_tests(
    Int4PlainInt32Tensor, globals(), only_for=("xpu", "npu"), allow_xpu=True
)


if __name__ == "__main__":
    run_tests()

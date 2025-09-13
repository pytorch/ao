# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao import quantize_
from torchao.quantization import PerGroup, PerRow, PerTensor
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.quantization.utils import compute_error
from torchao.utils import (
    torch_version_at_least,
)


def get_config(granularity):
    return Float8DynamicActivationFloat8WeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        granularity=granularity,
        packing_format="opaque",
    )


class ToyLinearModel(torch.nn.Module):
    def __init__(self, K=64, N=32, bias=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(K, N, bias=bias).to(torch.float)
        self.linear2 = torch.nn.Linear(N, K, bias=bias).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (
            torch.rand(batch_size, self.linear1.in_features, dtype=dtype, device=device)
            * 0.1,
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class TestDynamicFloat8Linear(TestCase):
    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    def test_dynamic_float8_linear_cpu(self, dtype, x_dim, bias, bs):
        device = "cpu"
        m = ToyLinearModel(256, 256, bias=bias).eval().to(dtype).to(device)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        with torch.no_grad():
            quantize_(
                m,
                get_config(PerRow()),
            )
            y1 = m(*example_inputs)
            assert compute_error(y, y1) > 20
            y2, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            assert compute_error(y, y2) > 20

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize(
        "granularity",
        [
            (PerTensor(), PerTensor()),
            (PerTensor(), PerRow()),
            (PerTensor(), PerGroup(64)),
        ],
    )
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 128])
    def test_dynamic_float8_linear_per_tensor_cpu(
        self, granularity, dtype, x_dim, bias, bs
    ):
        device = "cpu"
        m = ToyLinearModel(256, 256, bias=bias).eval().to(dtype).to(device)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        with torch.no_grad():
            quantize_(
                m,
                get_config(granularity),
            )
            y1 = m(*example_inputs)
            assert compute_error(y, y1) > 20
            y2, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            assert compute_error(y, y2) > 20

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    def test_dynamic_float8_linear_ref_cpu(self, dtype, x_dim, bias):
        device = "cpu"
        # the shape is not supported by cpp kernel, so the ref path will be used.
        m = ToyLinearModel(120, 120, bias=bias).eval().to(dtype).to(device)
        bs = 4
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        with torch.no_grad():
            quantize_(
                m,
                get_config(PerRow()),
            )
            y1 = m(*example_inputs)
            assert compute_error(y, y1) > 20
            y2, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            assert compute_error(y, y2) > 20

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    @common_utils.parametrize("group_size", [32, 64, 128])
    def test_dynamic_float8_linear_per_group_cpu(
        self, dtype, x_dim, bias, bs, group_size
    ):
        device = "cpu"
        m = ToyLinearModel(256, 256, bias=bias).eval().to(dtype).to(device)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        with torch.no_grad():
            quantize_(
                m,
                get_config([PerRow(), PerGroup(group_size)]),
            )
            y1 = m(*example_inputs)
            assert compute_error(y, y1) > 20
            y2, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            assert compute_error(y, y2) > 20

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    @common_utils.parametrize("group_size", [32, 64, 128])
    def test_dynamic_float8_linear_per_group_act_cpu(
        self, dtype, x_dim, bias, bs, group_size
    ):
        device = "cpu"
        m = ToyLinearModel(256, 256, bias=bias).eval().to(dtype).to(device)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        with torch.no_grad():
            quantize_(
                m,
                get_config([PerGroup(group_size), PerGroup(group_size)]),
            )
            y1 = m(*example_inputs)
            assert compute_error(y, y1) > 20
            y2, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            assert compute_error(y, y2) > 20

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @common_utils.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_module_path(self, dtype):
        linear = torch.nn.Linear(128, 256, dtype=dtype)
        quantize_(linear, get_config(PerRow()))
        self.assertEqual(
            str(type(linear.weight)),
            "<class 'torchao.quantization.Float8OpaqueTensor'>",
        )

        with tempfile.NamedTemporaryFile() as f:
            torch.save(linear.state_dict(), f)
            f.seek(0)
            state_dict = torch.load(f)
            self.assertEqual(
                str(type(state_dict["weight"])),
                "<class 'torchao.quantization.Float8OpaqueTensor'>",
            )


common_utils.instantiate_parametrized_tests(TestDynamicFloat8Linear)


if __name__ == "__main__":
    run_tests()

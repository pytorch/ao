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
from torchao.quantization import (
    Float8DynamicActivationFloat8WeightConfig,
    PerGroup,
    PerRow,
    PerTensor,
)
from torchao.quantization.utils import compute_error
from torchao.testing.model_architectures import ToyTwoLinearModel
from torchao.utils import (
    torch_version_at_least,
)


def get_config(granularity):
    return Float8DynamicActivationFloat8WeightConfig(
        activation_dtype=torch.float8_e4m3fn,
        granularity=granularity,
        float8_packing_format="opaque",
    )


@common_utils.instantiate_parametrized_tests
class TestFloat8OpaqueTensor(TestCase):
    """Test cases for Float8OpaqueTensor on CPU"""

    def setUp(self):
        torch.set_grad_enabled(False)

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::float8_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not torch_version_at_least("2.6.0"), "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    @common_utils.parametrize(
        "x_granularity",
        [PerTensor(), PerRow(), PerGroup(32), PerGroup(64), PerGroup(128)],
    )
    @common_utils.parametrize(
        "w_granularity",
        [PerTensor(), PerRow(), PerGroup(32), PerGroup(64), PerGroup(128)],
    )
    def test_dynamic_float8_linear(
        self, dtype, x_dim, bias, bs, x_granularity, w_granularity
    ):
        if isinstance(x_granularity, PerGroup):
            if not isinstance(w_granularity, PerGroup):
                return
            if w_granularity.group_size != x_granularity.group_size:
                return
        device = "cpu"
        m = ToyTwoLinearModel(256, 256, 256, dtype, device, bias).eval()
        example_inputs = m.example_inputs(batch_size=bs)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

        quantize_(
            m,
            get_config([x_granularity, w_granularity]),
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
    @common_utils.parametrize("bs", [4, 128])
    def test_dynamic_float8_linear_fallback_path(self, dtype, x_dim, bias, bs):
        """
        Test the fallback implementation with a shape that is not supported by the optimized kernel
        """
        device = "cpu"
        m = ToyTwoLinearModel(120, 120, 120, dtype, device, bias).eval()
        example_inputs = m.example_inputs(batch_size=bs)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)
        y = m(*example_inputs)

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


if __name__ == "__main__":
    run_tests()

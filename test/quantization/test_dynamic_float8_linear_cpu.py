# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import copy
import unittest

import torch
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

from torchao import quantize_
from torchao.dtypes import (
    Float8DynamicActFloat8WeightCPULayout,
    PlainLayout,
)
from torchao.quantization import PerRow
from torchao.quantization.quant_api import (
    Float8DynamicActivationFloat8WeightConfig,
)
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_6,
)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, K=64, N=32, bias=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(K, N, bias=bias).to(torch.float)
        self.linear2 = torch.nn.Linear(N, K, bias=bias).to(torch.float)

    def example_inputs(self, batch_size=1, dtype=torch.float, device="cpu"):
        return (
            torch.randn(
                batch_size, self.linear1.in_features, dtype=dtype, device=device
            ),
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
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_6, "Test only enabled for 2.6+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    def test_dynamic_float8_linear_cpu(self, dtype, x_dim, bias, bs):
        device = "cpu"
        m = ToyLinearModel(256, 256, bias=bias).eval().to(dtype).to(device)
        m2 = copy.deepcopy(m)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)

        with torch.no_grad():
            quantize_(
                m,
                Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                    layout=Float8DynamicActFloat8WeightCPULayout(),
                ),
            )
            y, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.float8_linear_cpu.default" in code[0]
            quantize_(
                m2,
                Float8DynamicActivationFloat8WeightConfig(
                    granularity=PerRow(),
                    layout=PlainLayout(),
                ),
            )
            torch._dynamo.reset()  # may segfault without this
            y2 = torch.compile(m2, fullgraph=True, dynamic=True)(*example_inputs)
            atol, rtol = 1e-6, 1e-6
            if dtype == torch.bfloat16:
                atol, rtol = 1.6e-2, 3e-3
            elif dtype == torch.half:
                atol, rtol = 6e-3, 2e-3
            assert torch.allclose(y, y2, atol=atol, rtol=rtol)
            # Test get_plain by dequantize()
            dqw1 = m.linear1.weight.original_weight_tensor.dequantize()
            dqw2 = m.linear2.weight.original_weight_tensor.dequantize()
            dqw1_ref = m2.linear1.weight.original_weight_tensor.dequantize()
            dqw2_ref = m2.linear2.weight.original_weight_tensor.dequantize()
            assert torch.allclose(dqw1, dqw1_ref)
            assert torch.allclose(dqw2, dqw2_ref)


common_utils.instantiate_parametrized_tests(TestDynamicFloat8Linear)


if __name__ == "__main__":
    run_tests()

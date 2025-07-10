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
    Int8DynamicActInt4WeightCPULayout,
    PlainLayout,
)
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt4WeightConfig,
)
from torchao.quantization.quant_primitives import MappingType
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_7,
    TORCH_VERSION_AT_LEAST_2_8,
)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64, bias=False):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=bias).to(torch.float)
        self.linear2 = torch.nn.Linear(n, k, bias=bias).to(torch.float)

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


class TestDa8w4Cpu(TestCase):
    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_7, "Test only enabled for 2.7+")
    @common_utils.parametrize("dtype", [torch.float, torch.bfloat16, torch.half])
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    @common_utils.parametrize("bs", [1, 160])
    @common_utils.parametrize("sym_quant_a", [True, False])
    def test_8da4w_cpu(self, dtype, x_dim, bias, bs, sym_quant_a):
        if sym_quant_a and not TORCH_VERSION_AT_LEAST_2_8:
            # not supported until PT 2.8
            return
        device = "cpu"
        m = ToyLinearModel(bias=bias).eval().to(dtype).to(device)
        m2 = copy.deepcopy(m)
        example_inputs = m.example_inputs(batch_size=bs, dtype=dtype, device=device)
        if x_dim == 3:
            example_inputs = (example_inputs[0].unsqueeze(0),)

        with torch.no_grad():
            # Currently, the difference between Int8DynamicActInt4WeightCPULayout and PlainLayout
            # is that the former packs two int4 weights into one int8, while the latter does not.
            quantize_(
                m,
                Int8DynamicActivationInt4WeightConfig(
                    group_size=32,
                    layout=Int8DynamicActInt4WeightCPULayout(),
                    act_mapping_type=MappingType.SYMMETRIC
                    if sym_quant_a
                    else MappingType.ASYMMETRIC,
                ),
            )
            y, code = torch._inductor.utils.run_and_get_code(
                torch.compile(m, fullgraph=True, dynamic=True),
                *example_inputs,
            )
            # ensure the expected op is in the code
            assert "torch.ops.torchao.da8w4_linear_cpu.default" in code[0]
            quantize_(
                m2,
                Int8DynamicActivationInt4WeightConfig(
                    group_size=32,
                    layout=PlainLayout(),
                    act_mapping_type=MappingType.SYMMETRIC
                    if sym_quant_a
                    else MappingType.ASYMMETRIC,
                ),
            )
            torch._dynamo.reset()  # may segfault without this
            y2 = torch.compile(m2, fullgraph=True, dynamic=True)(*example_inputs)
            atol, rtol = 4e-7, 1e-5
            if dtype == torch.bfloat16:
                atol, rtol = 1e-2, 3e-3
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

    @unittest.skipIf(
        "CPU" not in torch._C._dispatch_dump("torchao::da8w4_linear_cpu"),
        reason="cpp kernels not built",
    )
    @unittest.skipIf(not TORCH_VERSION_AT_LEAST_2_8, "Test only enabled for 2.8+")
    @common_utils.parametrize("x_dim", [2, 3])
    @common_utils.parametrize("bias", [True, False])
    def test_8da4w_concat_linear_cpu(self, x_dim, bias):
        N, K = 64, 128

        class Mod(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.linear1 = torch.nn.Linear(K, N, bias=bias)
                self.linear2 = torch.nn.Linear(K, N, bias=bias)
                self.linear3 = torch.nn.Linear(K, N, bias=bias)

            def forward(self, x):
                a = self.linear1(x)
                b = self.linear2(x)
                c = self.linear3(x)
                return a + b + c

        dtype = torch.bfloat16
        device = "cpu"
        m = Mod(bias).eval().to(dtype).to(device)
        x_shape = [2] * x_dim
        x_shape[-1] = K
        x = torch.rand(x_shape, dtype=dtype, device=device)
        with torch.no_grad():
            quantize_(
                m,
                Int8DynamicActivationInt4WeightConfig(
                    group_size=32,
                    layout=Int8DynamicActInt4WeightCPULayout(),
                    act_mapping_type=MappingType.SYMMETRIC,
                ),
            )
            # Need to turn on freezing to get the pattern
            # set enable_concat_linear to true to enable the fusion
            with torch._inductor.config.patch(
                {"freezing": True, "cpp.enable_concat_linear": True}
            ):
                y, code = torch._inductor.utils.run_and_get_code(
                    torch.compile(m, fullgraph=True, dynamic=True),
                    x,
                )
            # ensure the expected op occurs only once in the code after fusion
            # The trailing "(" is to avoid matching the op in the comment
            # assert code[0].count("torch.ops.torchao.da8w4_linear_cpu.default(") == 1
            # with torch._inductor.config.patch(
            #     {"freezing": True, "cpp.enable_concat_linear": False}
            # ):
            #     y_ref, code = torch._inductor.utils.run_and_get_code(
            #         torch.compile(m, fullgraph=True, dynamic=True),
            #         x,
            #     )
            # assert torch.allclose(y, y_ref)


common_utils.instantiate_parametrized_tests(TestDa8w4Cpu)


if __name__ == "__main__":
    run_tests()

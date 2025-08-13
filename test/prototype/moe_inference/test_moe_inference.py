# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the ESD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import pytest
import torch
import torch.nn as nn

from torchao.prototype.moe_inference.workflow import (
    convert_to_float8_moe_inference,
    print_param_quant_info,
)
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerRow
from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
)
from torchao.quantization.utils import compute_error
from torchao.testing.torchtitan_moe import MoE, MoEArgs
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)

if not TORCH_VERSION_AT_LEAST_2_8:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


torch.manual_seed(0)


class GroupedMMWrapper(nn.Module):
    def __init__(self, E, K, N):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(E, N, K))

    def forward(self, x, offs):
        wt = self.weight.transpose(-2, -1)  # E, N, K -> E, K, N
        y = torch._grouped_mm(x, wt, offs)
        return y


def _generate_test_offsets(num_tokens, num_experts):
    """
    Given `num_tokens` and `num_experts`, generates a tensor of offsets
    compatible with `torch._grouped_mm` and `torch._scaled_grouped_mm`.
    This is useful for mocking out real token routing.
    """
    # for now, return [1, ..., num_experts, ..., num_tokens]
    res = list(x + 1 for x in range(num_experts))
    res[-1] = num_tokens
    return torch.tensor(res, device="cuda", dtype=torch.int32)


class TestMoEInference:
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not is_sm_at_least_90(), "Requires CUDA capability >= 9.0")
    @torch.no_grad()
    def test_hello_world(self):
        M, E, K, N = 4, 2, 16, 32
        m = GroupedMMWrapper(E, K, N).cuda().bfloat16()
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        offs = _generate_test_offsets(M, E)
        y = m(x, offs)
        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        filter_fn = lambda param, param_name: param_name.endswith("weight")
        convert_to_float8_moe_inference(m, config, filter_fn)
        assert type(m.weight) == Float8Tensor
        yq = m(x, offs)
        sqnr = compute_error(y, yq).item()
        assert sqnr > 25.0

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    @unittest.skipIf(not is_sm_at_least_90(), "Requires CUDA capability >= 9.0")
    @torch.no_grad()
    def test_torchtitan_moe(self):
        moe_args = MoEArgs(num_experts=2)
        dim, hidden_dim = 512, 1024
        batch, seq, dim = 8, 2048, dim
        with torch.device("cuda"):
            moe = MoE(moe_args, dim, hidden_dim).to(torch.bfloat16)
            moe.init_weights(init_std=0.1, buffer_device="cuda")
            x = torch.randn(batch, seq, dim, dtype=torch.bfloat16)

        y_ref = moe(x)

        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
        filter_fn = (
            lambda p, n: n.endswith("experts.w1")
            or n.endswith("experts.w2")
            or n.endswith("experts.w3")
        )
        convert_to_float8_moe_inference(moe, config, filter_fn)

        assert type(moe.experts.w1) == Float8Tensor
        assert type(moe.experts.w2) == Float8Tensor
        assert type(moe.experts.w3) == Float8Tensor

        print_param_quant_info(moe)

        yq = moe(x)
        sqnr = compute_error(y_ref, yq).item()
        assert sqnr > 25.0


if __name__ == "__main__":
    pytest.main([__file__])

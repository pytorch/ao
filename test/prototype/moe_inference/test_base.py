# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import unittest

import pytest
import torch
import torch.nn as nn

from torchao.prototype.moe_inference.workflow import convert_to_float8_moe_inference
from torchao.quantization.utils import compute_error
from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_8,
    is_sm_at_least_90,
)

if not TORCH_VERSION_AT_LEAST_2_8:
    pytest.skip("Unsupported PyTorch version", allow_module_level=True)


torch.manual_seed(0)


class GroupedMMWrapper(nn.Module):
    def __init__(self, B, K, N):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(B, N, K))

    def forward(self, x, offs):
        wt = self.weight.transpose(-2, -1)  # B, N, K -> B, K, N
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
        M, B, K, N = 4, 2, 16, 32
        m = GroupedMMWrapper(B, K, N).cuda().bfloat16()
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        offs = _generate_test_offsets(M, B)
        y = m(x, offs)
        convert_to_float8_moe_inference(m, "weight")
        yq = m(x, offs)
        sqnr = compute_error(y, yq).item()
        assert sqnr > 25.0


if __name__ == "__main__":
    pytest.main([__file__])

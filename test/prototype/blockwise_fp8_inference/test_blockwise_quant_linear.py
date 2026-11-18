# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchao.prototype.blockwise_fp8_inference import blockwise_linear


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_blockwise_quant_linear_forwards_dtype_to_act_quant(dtype, monkeypatch):
    """
    Regression test for storing the constructor `dtype` on the module; forward()
    must pass it to fp8_blockwise_act_quant (see pytorch/ao#5070).
    """
    captured = []

    def fake_fp8_blockwise_act_quant(x, block_size, dt):
        captured.append(dt)
        return x, torch.zeros(1, dtype=torch.float32)

    def fake_blockwise_fp8_gemm(x, scale, weight, weight_scale, block_size):
        return torch.zeros(x.shape[0], weight.shape[0], dtype=torch.bfloat16)

    monkeypatch.setattr(
        blockwise_linear, "fp8_blockwise_act_quant", fake_fp8_blockwise_act_quant
    )
    monkeypatch.setattr(
        blockwise_linear, "_blockwise_fp8_gemm", fake_blockwise_fp8_gemm
    )

    layer = blockwise_linear.BlockwiseQuantLinear(16, 32, bias=False, dtype=dtype)
    assert layer.dtype is dtype

    x = torch.randn(2, 16)
    layer(x)

    assert captured == [dtype]

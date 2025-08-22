# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from packaging import version

from torchao.utils import auto_detect_device

_DEVICE = auto_detect_device()

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_inference.blockwise_quantization import (
    blockwise_fp8_gemm,
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_dequant,
    fp8_blockwise_weight_quant,
)
from torchao.utils import is_sm_at_least_89
from torchao.testing.utils import skip_if_xpu

BLOCKWISE_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


@pytest.mark.parametrize("_, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
def test_blockwise_quant_dequant(_, N, K, dtype):
    x = torch.randn(N, K).to(_DEVICE)
    qx, s = fp8_blockwise_weight_quant(x, dtype=dtype)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s)
    error = torch.norm(x - x_reconstructed) / torch.norm(x)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.1, "Quant-Dequant error is too high"


@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
@skip_if_xpu("XPU Enablement in Progress")
def test_blockwise_fp8_gemm(M, N, K, dtype):
    A = torch.randn(M, K).to(_DEVICE)
    B = torch.randn(N, K).to(_DEVICE)
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant(A, dtype=dtype)
    B_q, B_s = fp8_blockwise_weight_quant(B, dtype=dtype)
    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    error = torch.norm(C - C_q) / torch.norm(C)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.1, "Quantize gemm error is too high"

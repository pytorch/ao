# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.float8.config import e4m3_dtype
from torchao.kernel.blockwise_quantization import (
    blockwise_fp8_gemm,
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_dequant,
    fp8_blockwise_weight_quant,
)
from torchao.utils import is_MI300, is_MI350, is_sm_at_least_90

BLOCKWISE_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("_, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [e4m3_dtype])
def test_blockwise_quant_dequant(_, N, K, dtype):
    x = torch.randn(N, K).cuda()
    qx, s = fp8_blockwise_weight_quant(x, dtype=dtype)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s)
    error = torch.linalg.vector_norm(x - x_reconstructed) / torch.linalg.vector_norm(x)
    assert error < 0.1, f"Quant-Dequant error too high: {error.item():.6f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not (is_sm_at_least_90() or is_MI300() or is_MI350()),
    reason="Requires FP8-capable GPU (CUDA SM90+, MI300, or MI350)",
)
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [e4m3_dtype])
def test_blockwise_fp8_gemm(M, N, K, dtype):
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant(A, dtype=dtype)
    B_q, B_s = fp8_blockwise_weight_quant(B, dtype=dtype)
    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    assert C_q.dtype == torch.bfloat16, "unsupported"
    error = torch.linalg.vector_norm(C - C_q) / torch.linalg.vector_norm(C)
    assert error < 0.1, f"Quantize gemm error too high: {error.item():.6f}"

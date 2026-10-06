# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from packaging import version

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8_inference.kernels import fp8_blockwise_act_quant
from torchao.quantization.quantize_.workflows.float8.kernels import _blockwise_fp8_gemm
from torchao.utils import is_sm_at_least_90

BLOCKWISE_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


def _weight_quant_reference(w: torch.Tensor, block_size: int, dtype: torch.dtype):
    # Eager reference for block_size x block_size blockwise fp8 weight quant,
    # matching the semantics used by _blockwise_fp8_gemm's b/b_scale operands.
    n, k = w.shape
    wr = w.reshape(n // block_size, block_size, k // block_size, block_size)
    s = wr.abs().amax(dim=(1, 3), keepdim=True).float() / 448.0
    w_q = (wr / s).to(dtype).reshape(n, k)
    return w_q, s.reshape(n // block_size, k // block_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    version.parse(triton.__version__) < version.parse("3.3.0"),
    reason="Triton version < 3.3.0, test skipped",
)
@pytest.mark.skipif(not is_sm_at_least_90(), reason="Requires CUDA capability >= 9.0")
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_blockwise_fp8_gemm(M, N, K, dtype):
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant(A, dtype=dtype)
    B_q, B_s = _weight_quant_reference(B, block_size=128, dtype=dtype)
    C_q = _blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    assert C_q.dtype == torch.bfloat16, "unsupported"
    error = torch.linalg.vector_norm(C - C_q) / torch.linalg.vector_norm(C)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.1, "Quantize gemm error is too high"

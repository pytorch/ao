import pytest
import torch

triton = pytest.importorskip("triton", reason="Triton required to run this test")

from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm
from torchao.prototype.blockwise_fp8.blockwise_quantization import (
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_dequant,
    fp8_blockwise_weight_quant,
)
from torchao.utils import is_sm_at_least_89

BLOCKWISE_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("_, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
def test_blockwise_quant_dequant(_, N, K, dtype):
    x = torch.randn(N, K).cuda()
    qx, s = fp8_blockwise_weight_quant(x, dtype=dtype)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s)
    error = torch.norm(x - x_reconstructed) / torch.norm(x)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.1, "Quant-Dequant error is too high"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M, N, K", BLOCKWISE_SIZE_MNK)
@pytest.mark.parametrize(
    "dtype",
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if is_sm_at_least_89()
    else [torch.float8_e5m2],
)
def test_blockwise_fp8_gemm(M, N, K, dtype):
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()
    C = A @ B.T
    A_q, A_s = fp8_blockwise_act_quant(A, dtype=dtype)
    B_q, B_s = fp8_blockwise_weight_quant(B, dtype=dtype)
    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    error = torch.norm(C - C_q) / torch.norm(C)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.1, "Quantize gemm error is too high"

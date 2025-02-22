import pytest
import torch

from torchao.prototype.blockwise_fp8.blockwise_fp8_gemm_triton import blockwise_fp8_gemm
from torchao.prototype.blockwise_fp8.blockwise_quantization import (
    fp8_blockwise_act_quant,
    fp8_blockwise_weight_dequant,
    fp8_blockwise_weight_quant,
)

ROWWISE_SCALED_LINEAR_CUTLASS_SIZE_MNK = [
    (2, 512, 128),
    (3, 2048, 2048),
    (4, 3584, 640),
    (13, 8704, 8576),
    (26, 18944, 1664),
    (67, 6656, 1408),
]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("_, N, K", ROWWISE_SCALED_LINEAR_CUTLASS_SIZE_MNK)
def test_quant_dequant(_, N, K):
    x = torch.randn(N, K).cuda()
    qx, s = fp8_blockwise_weight_quant(x, block_size=128)
    x_reconstructed = fp8_blockwise_weight_dequant(qx, s, block_size=128)
    error = torch.norm(x - x_reconstructed) / torch.norm(x)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.05, "Quant-Dequant error is too high"

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("M, N, K", ROWWISE_SCALED_LINEAR_CUTLASS_SIZE_MNK)
def test_blockwise_fp8_gemm(M, N, K):
    A = torch.randn(M, K).cuda()
    B = torch.randn(N, K).cuda()

    C = A @ B.T

    A_q, A_s = fp8_blockwise_act_quant(A, block_size=128)
    B_q, B_s = fp8_blockwise_weight_quant(B, block_size=128)

    C_q = blockwise_fp8_gemm(A_q, A_s, B_q, B_s)
    print(C_q, C)
    error = torch.norm(C - C_q) / torch.norm(C)
    print(f"Relative Error: {error.item():.6f}")

    assert error < 0.05, "Quantize gemm error is too high"


# test_quant_dequant()
# test_blockwise_fp8_gemm()
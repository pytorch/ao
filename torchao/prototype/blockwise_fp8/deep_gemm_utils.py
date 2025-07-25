import sys

import torch

try:
    import deep_gemm
except ImportError:
    print("Please install deepgemm to use this feature")
    sys.exit(0)


def scaled_mm_deep_gemm_128_1_128_128(a, b, a_scale, b_scale):
    M, K = a.shape
    N, K = b.shape
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    deep_gemm.gemm_fp8_fp8_bf16_nt((a, a_scale), (b, b_scale), out=out)
    return out


def scaled_mm_deep_gemm_128_1_128_1(a, b, a_scale, b_scale):
    M, K = a.shape
    N, K = b.shape
    # Note: the results from `wgrad_gemm_fp8_fp8_fp32_nt` are **accumulated**
    # into this tensor. For now, we initialize with `zeros` to get correct
    # numerics in toy examples. For a real use case, this will need to pass
    # in the gradient tensor directly.
    out = torch.zeros((M, N), dtype=torch.float, device=a.device)
    deep_gemm.wgrad_gemm_fp8_fp8_fp32_nt((a, a_scale), (b, b_scale), out=out)
    return out

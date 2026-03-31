#!/usr/bin/env python3

import torch

from torchao.float8.float8_utils import compute_error
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_gemm_1x128_128x128,
)


def dequantize_lhs_block_1x128(a_q: torch.Tensor, a_s: torch.Tensor) -> torch.Tensor:
    return a_q.float() * a_s.repeat_interleave(128, dim=1)[:, : a_q.size(1)]


def dequantize_rhs_block_128x128(
    b_q: torch.Tensor, b_s: torch.Tensor
) -> torch.Tensor:
    scales = b_s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return b_q.float() * scales[: b_q.size(0), : b_q.size(1)]


def run_case(k: int, m: int = 256, n: int = 256) -> dict[str, float | int]:
    torch.manual_seed(0)
    weight = torch.nn.Linear(n, k, bias=False, device="cuda").weight.detach().contiguous()
    grad_output = torch.ones(m, k, device="cuda", dtype=torch.float32)

    grad_output_fp8, grad_output_scale = triton_fp8_blockwise_act_quant_lhs(
        grad_output, 128
    )
    weight_fp8, weight_scale = triton_fp8_blockwise_weight_quant_rhs(weight, 128)

    fp32_ref = grad_output @ weight
    dequant_ref = dequantize_lhs_block_1x128(
        grad_output_fp8, grad_output_scale
    ) @ dequantize_rhs_block_128x128(weight_fp8, weight_scale)
    scaled_mm_out = torch._scaled_mm(
        grad_output_fp8,
        weight_fp8,
        grad_output_scale,
        weight_scale,
        out_dtype=torch.bfloat16,
    )
    triton_out = triton_fp8_gemm_1x128_128x128(
        grad_output_fp8,
        weight_fp8,
        grad_output_scale,
        weight_scale,
        out_dtype=torch.bfloat16,
    )

    return {
        "k": k,
        "k_blocks": k // 128,
        "scaled_mm_sqnr_vs_fp32": float(compute_error(fp32_ref, scaled_mm_out)),
        "triton_sqnr_vs_fp32": float(compute_error(fp32_ref, triton_out)),
        "scaled_mm_sqnr_vs_dequant": float(compute_error(dequant_ref, scaled_mm_out)),
        "triton_sqnr_vs_dequant": float(compute_error(dequant_ref, triton_out)),
        "scaled_mm_norm": float(scaled_mm_out.float().norm()),
        "triton_norm": float(triton_out.float().norm()),
        "fp32_ref_norm": float(fp32_ref.norm()),
        "dequant_ref_norm": float(dequant_ref.norm()),
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    for k in (128000, 128128, 128256):
        result = run_case(k)
        print(result)


if __name__ == "__main__":
    main()

"""
Calculate FLOPs for the forward and backward pass of MXFP8GroupedExpertsFunc
(moe.py) as used in bench_moe_e2e.py.

Usage:
    python benchmarks/prototype/moe_training/mxfp8/ep/syncless/calc_flops.py

    # With custom dimensions:
    python benchmarks/prototype/moe_training/mxfp8/ep/syncless/calc_flops.py \
        --dim 7168 --hidden_dim 2048 --num_experts 8 --num_ranks 2 --seq_len 8192
"""

import argparse


def calc_moe_flops(
    dim: int,
    hidden_dim: int,
    num_experts: int,
    num_ranks: int,
    seq_len: int,
):
    local_experts = num_experts // num_ranks
    tokens_per_expert = seq_len // num_experts  # assumes load balance
    total_local_tokens = local_experts * tokens_per_expert
    T = tokens_per_expert  # tokens per expert group

    print("=== MoE FLOP Calculation ===")
    print(f"  dim (K)         = {dim}")
    print(f"  hidden_dim      = {hidden_dim}")
    print(f"  num_experts     = {num_experts}")
    print(f"  num_ranks       = {num_ranks}")
    print(f"  local_experts   = {local_experts}")
    print(f"  seq_len         = {seq_len}")
    print(f"  tokens/expert   = {T}")
    print(f"  total local T   = {total_local_tokens}")
    print()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================
    print("--- FORWARD PASS ---")

    # GEMM 1: h13 = x @ w13.T
    #   x:   (T, dim) per expert
    #   w13: (2*hidden_dim, dim) per expert  ->  transposed: (dim, 2*hidden_dim)
    #   FLOPs per expert: 2 * T * dim * (2 * hidden_dim)
    gemm1_per_expert = 2 * T * dim * (2 * hidden_dim)
    gemm1_total = local_experts * gemm1_per_expert
    print(
        f"  GEMM1 (x @ w13.T):  {gemm1_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {gemm1_total / 1e9:.3f} GFLOP"
    )

    # SwiGLU: silu(h1) * h3
    #   sigmoid: 4 ops (negate, exp, add 1, reciprocal)
    #   silu = x * sigmoid(x): 1 mul
    #   h1 * h3: 1 mul
    #   Total: ~6 FLOPs per element
    swiglu_fw_per_expert = 6 * T * hidden_dim
    swiglu_fw_total = local_experts * swiglu_fw_per_expert
    print(
        f"  SwiGLU forward:     {swiglu_fw_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {swiglu_fw_total / 1e9:.3f} GFLOP"
    )

    # GEMM 2: out = h @ w2.T
    #   h:  (T, hidden_dim) per expert
    #   w2: (dim, hidden_dim) per expert  ->  transposed: (hidden_dim, dim)
    #   FLOPs per expert: 2 * T * hidden_dim * dim
    gemm2_per_expert = 2 * T * hidden_dim * dim
    gemm2_total = local_experts * gemm2_per_expert
    print(
        f"  GEMM2 (h @ w2.T):   {gemm2_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {gemm2_total / 1e9:.3f} GFLOP"
    )

    fwd_total = gemm1_total + swiglu_fw_total + gemm2_total
    fwd_gemm_only = gemm1_total + gemm2_total
    print("  ---")
    print(f"  Forward GEMM FLOPs: {fwd_gemm_only / 1e9:.3f} GFLOP")
    print(
        f"  Forward total:      {fwd_total / 1e9:.3f} GFLOP ({fwd_total / 1e12:.4f} TFLOP)"
    )
    print()

    # =========================================================================
    # BACKWARD PASS
    # =========================================================================
    print("--- BACKWARD PASS ---")

    # 1. w2 dgrad: grad_h = grad_out @ w2
    #   grad_out: (T, dim) per expert
    #   w2:       (dim, hidden_dim) per expert
    #   FLOPs: 2 * T * dim * hidden_dim
    w2_dgrad_per_expert = 2 * T * dim * hidden_dim
    w2_dgrad_total = local_experts * w2_dgrad_per_expert
    print(
        f"  w2 dgrad (grad_out @ w2):     {w2_dgrad_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {w2_dgrad_total / 1e9:.3f} GFLOP"
    )

    # 2. SwiGLU backward: recompute h + compute grad_h1, grad_h3
    #   Recompute: sigmoid(h1), silu(h1), h = silu(h1)*h3  -> ~6 FLOPs/elem
    #   dsilu = sig + h1*sig*(1-sig)                        -> ~5 FLOPs/elem
    #   grad_h1 = grad_h * h3 * dsilu                      -> ~2 FLOPs/elem
    #   grad_h3 = grad_h * silu(h1)                        -> ~1 FLOP/elem
    #   Total: ~14 FLOPs per element
    swiglu_bw_per_expert = 14 * T * hidden_dim
    swiglu_bw_total = local_experts * swiglu_bw_per_expert
    print(
        f"  SwiGLU backward:              {swiglu_bw_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {swiglu_bw_total / 1e9:.3f} GFLOP"
    )

    # 3. w2 wgrad: grad_w2 = grad_out.T @ h
    #   grad_out.T: (dim, T) per expert
    #   h:          (T, hidden_dim) per expert
    #   FLOPs: 2 * dim * T * hidden_dim
    w2_wgrad_per_expert = 2 * dim * T * hidden_dim
    w2_wgrad_total = local_experts * w2_wgrad_per_expert
    print(
        f"  w2 wgrad (grad_out.T @ h):    {w2_wgrad_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {w2_wgrad_total / 1e9:.3f} GFLOP"
    )

    # 4. w13 dgrad: grad_x = grad_h13 @ w13
    #   grad_h13: (T, 2*hidden_dim) per expert
    #   w13:      (2*hidden_dim, dim) per expert
    #   FLOPs: 2 * T * (2*hidden_dim) * dim
    w13_dgrad_per_expert = 2 * T * (2 * hidden_dim) * dim
    w13_dgrad_total = local_experts * w13_dgrad_per_expert
    print(
        f"  w13 dgrad (grad_h13 @ w13):   {w13_dgrad_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {w13_dgrad_total / 1e9:.3f} GFLOP"
    )

    # 5. w13 wgrad: grad_w13 = grad_h13.T @ x
    #   grad_h13.T: (2*hidden_dim, T) per expert
    #   x:          (T, dim) per expert
    #   FLOPs: 2 * (2*hidden_dim) * T * dim
    w13_wgrad_per_expert = 2 * (2 * hidden_dim) * T * dim
    w13_wgrad_total = local_experts * w13_wgrad_per_expert
    print(
        f"  w13 wgrad (grad_h13.T @ x):   {w13_wgrad_per_expert / 1e9:.3f} GFLOP/expert  x {local_experts} = {w13_wgrad_total / 1e9:.3f} GFLOP"
    )

    bwd_total = (
        w2_dgrad_total
        + swiglu_bw_total
        + w2_wgrad_total
        + w13_dgrad_total
        + w13_wgrad_total
    )
    bwd_gemm_only = w2_dgrad_total + w2_wgrad_total + w13_dgrad_total + w13_wgrad_total
    print("  ---")
    print(f"  Backward GEMM FLOPs: {bwd_gemm_only / 1e9:.3f} GFLOP")
    print(
        f"  Backward total:      {bwd_total / 1e9:.3f} GFLOP ({bwd_total / 1e12:.4f} TFLOP)"
    )
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    total = fwd_total + bwd_total
    total_gemm = fwd_gemm_only + bwd_gemm_only
    print("--- SUMMARY ---")
    print(
        f"  Forward:           {fwd_total / 1e9:>10.3f} GFLOP  ({fwd_total / 1e12:.4f} TFLOP)"
    )
    print(
        f"  Backward:          {bwd_total / 1e9:>10.3f} GFLOP  ({bwd_total / 1e12:.4f} TFLOP)"
    )
    print(
        f"  Total (fwd+bwd):   {total / 1e9:>10.3f} GFLOP  ({total / 1e12:.4f} TFLOP)"
    )
    print(f"  Backward/Forward:  {bwd_total / fwd_total:.2f}x")
    print(
        f"  GEMM-only total:   {total_gemm / 1e9:>10.3f} GFLOP  ({total_gemm / total * 100:.1f}% of total)"
    )
    print()

    # Roofline — use specs from torchao roofline_utils
    from torchao.testing.training.roofline_utils import gpu_name_to_specs

    print("--- ROOFLINE (compute-bound lower bound, dense) ---")
    b200 = gpu_name_to_specs["NVIDIA B200"]
    for label, peak in [
        ("B200 BF16", b200["bf16_peak_tops"]),
        ("B200 FP8", b200["fp8_peak_tops"]),
    ]:
        fwd_ms = fwd_gemm_only / peak * 1e3
        bwd_ms = bwd_gemm_only / peak * 1e3
        total_ms = total_gemm / peak * 1e3
        print(
            f"  {label:20s}: fwd={fwd_ms:.3f}ms  bwd={bwd_ms:.3f}ms  total={total_ms:.3f}ms"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MoE FLOPs")
    parser.add_argument("--dim", type=int, default=7168, help="Model dimension")
    parser.add_argument(
        "--hidden_dim", type=int, default=2048, help="Expert hidden dimension"
    )
    parser.add_argument(
        "--num_experts", type=int, default=8, help="Total number of experts"
    )
    parser.add_argument("--num_ranks", type=int, default=2, help="Number of EP ranks")
    parser.add_argument(
        "--seq_len", type=int, default=8192, help="Sequence length (total tokens)"
    )
    args = parser.parse_args()

    calc_moe_flops(
        dim=args.dim,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        num_ranks=args.num_ranks,
        seq_len=args.seq_len,
    )

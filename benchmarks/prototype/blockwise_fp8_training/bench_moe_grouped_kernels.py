# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
"""
Per-kernel benchmark for the blockwise FP8 MoE grouped-GEMM path.

Mirrors the linear blockwise FP8 benchmarks: each kernel the MoE op dispatches
is timed in isolation. Quantization (cast) kernels are memory-bound and reported
against the memory-bandwidth roofline (GB/s, % of achievable). The DeepGEMM
grouped GEMMs are compute-bound and reported against the FP8 tensor-core
roofline (TFLOP/s, % of achievable); their modeled memory traffic is also shown
so memory-bound cases (e.g. the K-grouped wgrad, which reads+writes FP32
accumulators) are visible.

Byte accounting for the cast kernels counts the high-precision input read plus
the FP8 data and FP32 scale writes (the actual operand tensors produced).

Kernels timed, matching torchao.prototype.moe_training.blockwise_fp8.grouped_mm:

  forward
    - act_quant_lhs                 (activations -> FP8 1x128)
    - weight_quant_forward_rhs      (weights -> DeepGEMM (E,N,K) FP8 128x128)
    - deepgemm_grouped_mm           (out = A @ B_t)
  backward (dgrad)
    - act_quant_lhs(grad_out)
    - weight_quant_dgrad_rhs        (weights -> DeepGEMM (E,K,N) FP8 128x128)
    - deepgemm_grouped_mm_dgrad     (grad_A = grad_out @ B)
  backward (wgrad)
    - wgrad_quant_lhs               (K-grouped activation quant of grad_out)
    - wgrad_quant_rhs               (K-grouped activation quant of A)
    - deepgemm_grouped_mm_wgrad     (grad_B = grad_out^T @ A, K-grouped)
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch
from tabulate import tabulate
from triton.testing import do_bench

from torchao.float8.config import e4m3_dtype
from torchao.prototype.blockwise_fp8_training.deepgemm_grouped_kernels import (
    _quantize_wgrad_lhs,
    _quantize_wgrad_rhs,
    _should_quantize_k_grouped_directly,
    deepgemm_blockwise_scaled_grouped_mm,
    deepgemm_blockwise_scaled_grouped_mm_wgrad,
    is_deep_gemm_available,
    prepare_deepgemm_wgrad_plan,
)
from torchao.prototype.blockwise_fp8_training.deepgemm_metadata import (
    build_deepgemm_grouped_offset_plan,
)
from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
    triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs,
    triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.testing.training.roofline_utils import gpu_name_to_specs

device = torch.device("cuda")


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs) -> float:
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


class Kind:
    MEM = "mem"  # memory-bound cast kernel -> GB/s
    GEMM = "gemm"  # compute-bound GEMM -> TFLOP/s (+ modeled mem traffic)


@dataclass
class KernelMeasurement:
    name: str
    kind: str
    us: float
    bytes_moved: int
    flops: float


@dataclass(frozen=True)
class Shape:
    M: int
    N: int
    K: int
    E: int


def _lookup_specs(gpu_name: str) -> Optional[dict]:
    specs = gpu_name_to_specs.get(gpu_name)
    if specs is not None:
        return specs
    for known, candidate in gpu_name_to_specs.items():
        if known in gpu_name or gpu_name in known:
            return candidate
    return None


def _peak_mem_bw_from_device_properties() -> Optional[float]:
    # Matches benchmark_quant_kernel_bandwidth.py: prefer the real device HBM
    # peak over the roofline_utils value (which is a Meta-specific H100 variant).
    props = torch.cuda.get_device_properties(0)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None
    return (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0


def _io_bytes(*tensors: torch.Tensor) -> int:
    return sum(t.numel() * t.element_size() for t in tensors)


def _make_offsets(E: int, M: int, block_size: int, jagged: bool) -> torch.Tensor:
    if jagged:
        # Skewed per-expert token counts (real routing). Stresses load balance,
        # so the grouped GEMM rooflines reflect distribution, not just the kernel.
        return generate_jagged_offs(E, M, multiple_of=block_size, device=device)
    # Balanced: equal tokens per expert. Isolates kernel efficiency from skew.
    assert M % E == 0, "balanced offsets require M divisible by E"
    toks = M // E
    assert toks % block_size == 0, "balanced per-expert tokens must be block-aligned"
    return torch.arange(toks, M + 1, toks, dtype=torch.int32, device=device)


def _bench_shape(
    shape: Shape, block_size: int, jagged: bool
) -> List[KernelMeasurement]:
    M, N, K, E = shape.M, shape.N, shape.K, shape.E
    out_dtype = torch.bfloat16
    fp8 = e4m3_dtype
    recipe_a = _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE)
    recipe_b = _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE)

    # Inputs in the layouts the MoE op uses.
    A = torch.randn(M, K, dtype=out_dtype, device=device)
    grad_out = torch.randn(M, N, dtype=out_dtype, device=device)
    # B_t logical (E, K, N) in per-expert column-major layout.
    weight = torch.randn(E, N, K, dtype=out_dtype, device=device)
    B_t = weight.contiguous().transpose(-2, -1)

    offs = _make_offsets(E, M, block_size, jagged)
    offset_plan = build_deepgemm_grouped_offset_plan(offs, num_rows=M)
    # Touch the cached host-side group sizes once, outside every timed region,
    # so the K-grouped quant/GEMM kernels are not charged for the D2H sync.
    group_sizes = offset_plan.group_sizes

    measurements: List[KernelMeasurement] = []

    def time_mem(name, fn, in_bytes, out_data, out_scale):
        us = benchmark_cuda_function_in_microseconds(fn)
        moved = in_bytes + _io_bytes(out_data, out_scale)
        measurements.append(KernelMeasurement(name, Kind.MEM, us, moved, 0.0))

    def time_gemm(name, fn, flops, bytes_moved):
        us = benchmark_cuda_function_in_microseconds(fn)
        measurements.append(KernelMeasurement(name, Kind.GEMM, us, bytes_moved, flops))

    # ---- forward ----
    A_fp8, A_scale = triton_fp8_blockwise_act_quant_lhs(
        A.contiguous(), block_size=block_size, dtype=fp8
    )
    time_mem(
        "fwd: act_quant_lhs",
        lambda: triton_fp8_blockwise_act_quant_lhs(
            A.contiguous(), block_size=block_size, dtype=fp8
        ),
        _io_bytes(A),
        A_fp8,
        A_scale,
    )

    B_fwd_fp8, B_fwd_scale = triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
        B_t, block_size=block_size, dtype=fp8
    )
    time_mem(
        "fwd: weight_quant_forward_rhs",
        lambda: triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
            B_t, block_size=block_size, dtype=fp8
        ),
        _io_bytes(B_t),
        B_fwd_fp8,
        B_fwd_scale,
    )

    # GEMM mem traffic: read A_fp8 + scales + B_fp8 + scales, write bf16 out.
    fwd_gemm_bytes = (
        _io_bytes(A_fp8, A_scale, B_fwd_fp8, B_fwd_scale) + M * N * 2  # bf16 out
    )
    time_gemm(
        "fwd: deepgemm_grouped_mm",
        lambda: deepgemm_blockwise_scaled_grouped_mm(
            A_fp8,
            B_fwd_fp8,
            A_scale,
            recipe_a,
            B_fwd_scale,
            recipe_b,
            offs,
            out_dtype,
            block_size,
            offset_plan=offset_plan,
        ),
        2.0 * M * N * K,
        fwd_gemm_bytes,
    )

    # ---- backward: dgrad ----
    gout_fp8, gout_scale = triton_fp8_blockwise_act_quant_lhs(
        grad_out.contiguous(), block_size=block_size, dtype=fp8
    )
    time_mem(
        "bwd: act_quant_lhs(grad_out)",
        lambda: triton_fp8_blockwise_act_quant_lhs(
            grad_out.contiguous(), block_size=block_size, dtype=fp8
        ),
        _io_bytes(grad_out),
        gout_fp8,
        gout_scale,
    )

    B_dgrad_fp8, B_dgrad_scale = triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(
        B_t, block_size=block_size, dtype=fp8
    )
    time_mem(
        "bwd: weight_quant_dgrad_rhs",
        lambda: triton_fp8_blockwise_weight_quant_grouped_dgrad_rhs(
            B_t, block_size=block_size, dtype=fp8
        ),
        _io_bytes(B_t),
        B_dgrad_fp8,
        B_dgrad_scale,
    )

    dgrad_gemm_bytes = (
        _io_bytes(gout_fp8, gout_scale, B_dgrad_fp8, B_dgrad_scale) + M * K * 2
    )
    time_gemm(
        "bwd: deepgemm_grouped_mm_dgrad",
        lambda: deepgemm_blockwise_scaled_grouped_mm(
            gout_fp8,
            B_dgrad_fp8,
            gout_scale,
            recipe_a,
            B_dgrad_scale,
            recipe_b,
            offs,
            out_dtype,
            block_size,
            offset_plan=offset_plan,
        ),
        2.0 * M * N * K,
        dgrad_gemm_bytes,
    )

    # ---- backward: wgrad (K-grouped) ----
    # Build the per-block quant metadata once, outside the timed region (it is a
    # host-side python loop, not a kernel), so each quant row times only its
    # kernel. Each operand picks the direct K-grouped quant for wide dims and
    # TorchAO's transposed quant otherwise (see _DEEPGEMM_DIRECT..._MIN_DIM).
    lhs_md = (
        offset_plan.k_quant_metadata(block_size, N)
        if _should_quantize_k_grouped_directly(N)
        else None
    )
    rhs_md = (
        offset_plan.k_quant_metadata(block_size, K)
        if _should_quantize_k_grouped_directly(K)
        else None
    )
    lhs_op = _quantize_wgrad_lhs(
        grad_out, offset_plan.group_end_offsets, group_sizes, block_size, fp8, lhs_md
    )
    rhs_op = _quantize_wgrad_rhs(
        A, offset_plan.group_end_offsets, group_sizes, block_size, fp8, rhs_md
    )
    lhs_path = "direct" if _should_quantize_k_grouped_directly(N) else "transposed"
    rhs_path = "direct" if _should_quantize_k_grouped_directly(K) else "transposed"
    time_mem(
        f"bwd: wgrad_quant_lhs(grad_out) [{lhs_path}]",
        lambda: _quantize_wgrad_lhs(
            grad_out,
            offset_plan.group_end_offsets,
            group_sizes,
            block_size,
            fp8,
            lhs_md,
        ),
        _io_bytes(grad_out),
        lhs_op.data,
        lhs_op.scale,
    )
    time_mem(
        f"bwd: wgrad_quant_rhs(A) [{rhs_path}]",
        lambda: _quantize_wgrad_rhs(
            A, offset_plan.group_end_offsets, group_sizes, block_size, fp8, rhs_md
        ),
        _io_bytes(A),
        rhs_op.data,
        rhs_op.scale,
    )

    wgrad_plan = prepare_deepgemm_wgrad_plan(grad_out, A, offset_plan, block_size, fp8)
    assert wgrad_plan is not None, "wgrad plan requires block-aligned groups"
    # wgrad mem traffic: read lhs + rhs fp8 data + scales, read FP32 accum seed,
    # write FP32 (E,N,K) output. The two FP32 (E,N,K) buffers dominate.
    wgrad_gemm_bytes = (
        _io_bytes(
            wgrad_plan.lhs.data,
            wgrad_plan.lhs.scale,
            wgrad_plan.rhs.data,
            wgrad_plan.rhs.scale,
        )
        + 2 * E * N * K * 4  # FP32 accum read + FP32 out write
    )
    time_gemm(
        "bwd: deepgemm_grouped_mm_wgrad",
        lambda: deepgemm_blockwise_scaled_grouped_mm_wgrad(
            wgrad_plan.lhs,
            wgrad_plan.rhs,
            offset_plan,
            out_dtype,
            block_size,
        ),
        2.0 * M * N * K,
        wgrad_gemm_bytes,
    )

    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="+",
        # M, N, K, E. DeepSeek-V3 MoE FFN dims at a couple of token counts.
        default=[
            "16384,2048,7168,8",
            "32768,2048,7168,8",
            "16384,7168,2048,8",
            "32768,7168,2048,8",
        ],
        help="Comma-separated M,N,K,E groups.",
    )
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument(
        "--jagged",
        action="store_true",
        help="Use skewed per-expert token counts instead of balanced (default).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not is_deep_gemm_available():
        raise RuntimeError(
            "DeepGEMM is not importable; this benchmark targets the DeepGEMM backend."
        )

    gpu_name = torch.cuda.get_device_name(0)
    specs = _lookup_specs(gpu_name)
    if specs is None:
        raise RuntimeError(f"No roofline specs for GPU: {gpu_name}")

    device_peak_bw = _peak_mem_bw_from_device_properties()
    peak_bw = device_peak_bw or specs["peak_mem_bw_bytes_sec"]
    bw_source = "cuda_device_properties" if device_peak_bw else "roofline_utils"
    ach_bw = peak_bw * specs.get("pct_achievable_mem_bw", 1.0)
    peak_tops = specs["fp8_peak_tops"]
    ach_tops = peak_tops * specs.get("pct_achievable_gemm_tops", 1.0)

    print(f"GPU: {gpu_name}")
    print(
        f"Mem BW: peak {peak_bw / 1e9:.0f} GB/s (source: {bw_source}), "
        f"achievable {ach_bw / 1e9:.0f} GB/s "
        f"({specs.get('pct_achievable_mem_bw', 1.0) * 100:.0f}% of peak)"
    )
    print(
        f"FP8 compute: peak {peak_tops / 1e12:.0f} TFLOP/s, "
        f"achievable {ach_tops / 1e12:.0f} TFLOP/s "
        f"({specs.get('pct_achievable_gemm_tops', 1.0) * 100:.0f}% of peak)"
    )
    print(
        f"Tokens: {'jagged (skewed)' if args.jagged else 'balanced'}; "
        "128-aligned (no padding); DeepGEMM backend.\n"
    )

    torch.manual_seed(123)
    for shape_str in args.shapes:
        M, N, K, E = (int(x) for x in shape_str.split(","))
        shape = Shape(M, N, K, E)
        measurements = _bench_shape(shape, args.block_size, args.jagged)

        print(f"=== M={M} N={N} K={K} E={E} ===")
        rows = []
        for m in measurements:
            gbps = m.bytes_moved / 1e9 / (m.us * 1e-6)
            bw_pct = 100.0 * gbps * 1e9 / ach_bw
            if m.kind == Kind.MEM:
                rows.append(
                    [
                        m.name,
                        f"{m.us:.1f}",
                        "-",
                        "-",
                        f"{gbps:.0f}",
                        f"{bw_pct:.1f}",
                    ]
                )
            else:
                tflops = m.flops / 1e12 / (m.us * 1e-6)
                compute_pct = 100.0 * tflops * 1e12 / ach_tops
                rows.append(
                    [
                        m.name,
                        f"{m.us:.1f}",
                        f"{tflops:.0f}",
                        f"{compute_pct:.1f}",
                        f"{gbps:.0f}",
                        f"{bw_pct:.1f}",
                    ]
                )
        print(
            tabulate(
                rows,
                headers=[
                    "kernel",
                    "us",
                    "TFLOP/s",
                    "%ach_compute",
                    "GB/s",
                    "%ach_bw",
                ],
                tablefmt="github",
            )
        )
        print()


if __name__ == "__main__":
    main()

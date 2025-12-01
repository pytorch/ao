# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified roofline script for MXFP8 grouped GEMM and quantization kernels.

This script provides a centralized view of:
1. Net speedup: BF16 vs MXFP8 for forward + backward pass
2. Individual quantization kernel bandwidth utilization
"""

import fire
import matplotlib.pyplot as plt
import pandas as pd
import torch
from triton.testing import do_bench

from torchao.prototype.moe_training.kernels.mxfp8 import (
    torch_to_blocked_2d_M_groups,
    torch_to_blocked_per_group_3d,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_2d_M_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_cuda_3d,
)
from torchao.prototype.moe_training.scaled_grouped_mm import (
    ScaleCalculationMode as MoEScaleCalculationMode,
)
from torchao.prototype.moe_training.scaled_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.testing.training.roofline_utils import (
    gpu_name_to_specs,
)


# Compile to_mx wrapper function once at module level for reuse
def _to_mx_wrapper(data_hp, block_size, scaling_mode):
    """Wrapper function for to_mx that matches cast_bench.py pattern"""
    scale, data = to_mx(data_hp, torch.float8_e4m3fn, block_size, scaling_mode)
    return data, scale


_compiled_to_mx_wrapper = torch.compile(_to_mx_wrapper, fullgraph=True)


class RooflineModel:
    """Roofline model for grouped GEMM on B200 GPU"""

    def __init__(self, gpu_name="NVIDIA B200", power_limit_percent=100.0):
        """
        Args:
            gpu_name: GPU model name
            power_limit_percent: Power limit as percentage (0-100). Default 100.0
        """
        power_multiplier = power_limit_percent / 100.0

        if gpu_name in gpu_name_to_specs:
            self.gpu_specs = gpu_name_to_specs[gpu_name]
            self.bf16_tflops = (
                (self.gpu_specs["bf16_peak_tops"] / 1e12)
                * self.gpu_specs["pct_achievable_gemm_tops"]
                * power_multiplier
            )
            self.mxfp8_tflops = (
                (self.gpu_specs["fp8_peak_tops"] / 1e12)
                * self.gpu_specs["pct_achievable_gemm_tops"]
                * power_multiplier
            )
            self.memory_bandwidth_gbs = (
                (self.gpu_specs["peak_mem_bw_bytes_sec"] / 1e9)
                * self.gpu_specs["pct_achievable_mem_bw"]
                * power_multiplier
            )
        else:
            raise ValueError(f"Unsupported GPU: {gpu_name}")

    def compute_bf16_2d_3d_gemm_flops(self, M, K, N):
        """
        Compute FLOPs for BF16 2D-3D grouped GEMM (forward/backward input).

        Operation: (M, K) @ (G, K, N)^T -> (M, N)
        Each of M tokens goes to exactly one group.
        Total FLOPs = 2 * M * K * N
        """
        return 2 * M * K * N

    def compute_bf16_2d_2d_gemm_flops(self, N, M, K):
        """
        Compute FLOPs for BF16 2D-2D grouped GEMM (backward weight).

        Operation: (N, M) @ (M, K) -> G separate (N, K) matrices
        Each of M tokens contributes to exactly one group's gradient.
        Total FLOPs = 2 * N * M * K
        """
        return 2 * N * M * K

    def compute_bf16_fwd_bwd_time(self, M, K, N, G):
        """Compute time for BF16 forward + backward pass"""
        # Forward: (M, K) @ (G, K, N)^T -> (M, N)
        fwd_flops = self.compute_bf16_2d_3d_gemm_flops(M, K, N)

        # Backward input: (M, N) @ (G, N, K) -> (M, K)
        bwd_input_flops = self.compute_bf16_2d_3d_gemm_flops(M, N, K)

        # Backward weight: (N, M) @ (M, K) -> G separate (N, K)
        bwd_weight_flops = self.compute_bf16_2d_2d_gemm_flops(N, M, K)

        total_flops = fwd_flops + bwd_input_flops + bwd_weight_flops
        total_tflops = total_flops / 1e12
        time_s = total_tflops / self.bf16_tflops

        return time_s

    def compute_mxfp8_fwd_quant_time(self, M, K, G, N):
        """Compute time for quantizing inputs to MXFP8 for forward pass"""
        block_size = 32

        # Input quantization: (M,K)
        input_read_bytes = M * K * 2  # BF16
        input_write_bytes = M * K * 1 + M * (K // block_size) * 1

        # Weight quantization: (G,K,N)
        weight_read_bytes = G * K * N * 2  # BF16
        weight_write_bytes = G * K * N * 1 + G * N * (K // block_size) * 1

        total_bytes = (
            input_read_bytes
            + input_write_bytes
            + weight_read_bytes
            + weight_write_bytes
        )
        total_gb = total_bytes / 1e9
        time_s = total_gb / self.memory_bandwidth_gbs

        return time_s

    def compute_mxfp8_fwd_input_quant_time(self, M, K):
        """Compute time for quantizing input for forward pass"""
        block_size = 32
        read_bytes = M * K * 2  # BF16
        write_bytes = M * K * 1 + M * (K // block_size) * 1
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        return total_gb / self.memory_bandwidth_gbs

    def compute_mxfp8_fwd_weight_quant_time(self, G, K, N):
        """Compute time for quantizing weight for forward pass"""
        block_size = 32
        read_bytes = G * K * N * 2  # BF16
        write_bytes = G * K * N * 1 + G * N * (K // block_size) * 1
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        return total_gb / self.memory_bandwidth_gbs

    def compute_mxfp8_bwd_input_quant_time(self, M, K, G, N):
        """Compute time for quantizing inputs for backward pass (grad_input)"""
        block_size = 32

        # grad_output quantization: (M, N)
        grad_output_read_bytes = M * N * 2  # BF16
        grad_output_write_bytes = M * N * 1 + M * (N // block_size) * 1

        # weight quantization: (G, N, K)
        weight_read_bytes = G * N * K * 2  # BF16
        weight_write_bytes = G * N * K * 1 + G * K * (N // block_size) * 1

        total_bytes = (
            grad_output_read_bytes
            + grad_output_write_bytes
            + weight_read_bytes
            + weight_write_bytes
        )
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_mxfp8_bwd_weight_quant_time(self, M, K, G, N):
        """Compute time for quantizing inputs for backward pass (grad_weight)"""
        block_size = 32

        # grad_output.T quantization: (N, M)
        # grad_output has shape (M, N), transposed is (N, M)
        grad_output_t_read_bytes = N * M * 2  # BF16
        grad_output_t_write_bytes = N * M * 1 + N * (M // block_size) * 1

        # input quantization: (M, K)
        input_read_bytes = M * K * 2  # BF16
        input_write_bytes = M * K * 1 + (M // block_size) * K * 1

        total_bytes = (
            grad_output_t_read_bytes
            + grad_output_t_write_bytes
            + input_read_bytes
            + input_write_bytes
        )
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_mxfp8_2d_3d_gemm_flops(self, M, K, N):
        """
        Compute FLOPs for MXFP8 2D-3D grouped GEMM (forward/backward input).

        Operation: (M, K) @ (G, K, N)^T -> (M, N)
        Each of M tokens goes to exactly one group.
        Total FLOPs = 2 * M * K * N
        """
        return 2 * M * K * N

    def compute_mxfp8_2d_2d_gemm_flops(self, N, M, K):
        """
        Compute FLOPs for MXFP8 2D-2D grouped GEMM (backward weight).

        Operation: (N, M) @ (M, K) -> G separate (N, K) matrices
        G instances of (N, M/g) @ (M/g, K) -> G separate (N, K) matrices
        Total FLOPs = 2 * N * M * K
        """
        return 2 * N * M * K

    def compute_mxfp8_2d_3d_gemm_time(self, M, K, N):
        """Compute time for MXFP8 2D-3D grouped GEMM"""
        total_flops = self.compute_mxfp8_2d_3d_gemm_flops(M, K, N)
        total_tflops = total_flops / 1e12
        time_s = total_tflops / self.mxfp8_tflops
        return time_s

    def compute_mxfp8_2d_2d_gemm_time(self, N, M, K):
        """Compute time for MXFP8 2D-2D grouped GEMM"""
        total_flops = self.compute_mxfp8_2d_2d_gemm_flops(N, M, K)
        total_tflops = total_flops / 1e12
        time_s = total_tflops / self.mxfp8_tflops
        return time_s

    def compute_mxfp8_fwd_bwd_time(self, M, K, N, G):
        """Compute time for MXFP8 forward + backward pass including scale rearrangement overhead"""
        block_size = 32

        # Forward: (M, K) @ (G, K, N)^T -> (M, N) [2D-3D]
        fwd_quant_time = self.compute_mxfp8_fwd_quant_time(M, K, G, N)
        # Forward scale rearrangement:
        # - Input scales (M, K//32) -> M-groups rearrangement
        # - Weight scales (G, N, K//32) -> 3D per-group rearrangement
        fwd_input_scale_rearrange_time = self.compute_rearrange_2d_M_groups_time(
            M, K // block_size
        )
        fwd_weight_scale_rearrange_time = self.compute_rearrange_3d_per_group_time(
            G, N, K // block_size
        )
        fwd_gemm_time = self.compute_mxfp8_2d_3d_gemm_time(M, K, N)

        # Backward input: (M, N) @ (G, N, K) -> (M, K) [2D-3D]
        bwd_input_quant_time = self.compute_mxfp8_bwd_input_quant_time(M, K, G, N)
        # Backward input scale rearrangement:
        # - grad_output scales (M, N//32) -> M-groups rearrangement
        # - Weight scales (G, K, N//32) -> 3D per-group rearrangement (transposed weight)
        bwd_input_grad_scale_rearrange_time = self.compute_rearrange_2d_M_groups_time(
            M, N // block_size
        )
        bwd_input_weight_scale_rearrange_time = (
            self.compute_rearrange_3d_per_group_time(G, K, N // block_size)
        )
        bwd_input_gemm_time = self.compute_mxfp8_2d_3d_gemm_time(M, N, K)

        # Backward weight: (N, M) @ (M, K) -> G separate (N, K) [2D-2D]
        bwd_weight_quant_time = self.compute_mxfp8_bwd_weight_quant_time(M, K, G, N)
        # Backward weight scale rearrangement:
        # - grad_output.T scales (N, M//32) -> K-groups rearrangement
        # - Input scales (K, M//32) -> K-groups rearrangement
        bwd_weight_grad_scale_rearrange_time = self.compute_rearrange_2d_K_groups_time(
            N, M // block_size
        )
        bwd_weight_input_scale_rearrange_time = self.compute_rearrange_2d_K_groups_time(
            K, M // block_size
        )
        bwd_weight_gemm_time = self.compute_mxfp8_2d_2d_gemm_time(N, M, K)

        total_time = (
            fwd_quant_time
            + fwd_input_scale_rearrange_time
            + fwd_weight_scale_rearrange_time
            + fwd_gemm_time
            + bwd_input_quant_time
            + bwd_input_grad_scale_rearrange_time
            + bwd_input_weight_scale_rearrange_time
            + bwd_input_gemm_time
            + bwd_weight_quant_time
            + bwd_weight_grad_scale_rearrange_time
            + bwd_weight_input_scale_rearrange_time
            + bwd_weight_gemm_time
        )

        return total_time, fwd_quant_time, fwd_gemm_time

    def compute_speedup(self, M, K, N, G):
        """Compute speedup of MXFP8 vs BF16 for forward + backward pass"""
        bf16_time = self.compute_bf16_fwd_bwd_time(M, K, N, G)

        (
            mxfp8_total_time,
            mxfp8_fwd_quant_time,
            mxfp8_fwd_gemm_time,
        ) = self.compute_mxfp8_fwd_bwd_time(M, K, N, G)

        speedup = bf16_time / mxfp8_total_time

        return {
            "bf16_roofline_time_ms": bf16_time * 1000,
            "mxfp8_roofline_quant_time_ms": mxfp8_fwd_quant_time * 1000,
            "mxfp8_roofline_gemm_time_ms": mxfp8_fwd_gemm_time * 1000,
            "mxfp8_roofline_total_time_ms": mxfp8_total_time * 1000,
            "roofline_speedup": speedup,
        }

    def compute_quant_2d_time(self, M, K, block_size=32):
        """Compute roofline time for 2D quantization"""
        read_bytes = M * K * 2  # BF16
        write_bytes = M * K * 1 + M * (K // block_size) * 1  # FP8 + scales

        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_quant_3d_time(self, E, N, K, block_size=32):
        """Compute roofline time for 3D quantization"""
        read_bytes = E * N * K * 2  # BF16
        write_bytes = E * N * K * 1 + E * N * (K // block_size) * 1  # FP8 + scales

        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_rearrange_2d_M_groups_time(self, Mg, K):
        """
        Compute roofline time for 2D M-groups scale rearrangement.

        Args:
            Mg: Total number of tokens across all groups
            K: Number of scale blocks along K dimension

        Returns:
            Time in seconds
        """
        # Input: (Mg, K) uint8 scales
        read_bytes = Mg * K * 1  # uint8

        # Output: Rearranged (Mg, K) float8 scales
        write_bytes = Mg * K * 1  # float8

        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_rearrange_2d_K_groups_time(self, N, M):
        """
        Compute roofline time for 2D K-groups scale rearrangement.

        Args:
            N: Output dimension
            M: Total number of scale blocks

        Returns:
            Time in seconds
        """
        # Input: (N, M) uint8 scales
        read_bytes = N * M * 1  # uint8

        # Output: Rearranged (N, M) float8 scales
        write_bytes = N * M * 1  # float8

        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s

    def compute_rearrange_3d_per_group_time(self, G, N, K_blocks):
        """
        Compute roofline time for 3D per-group scale rearrangement.

        Args:
            G: Number of groups
            N: Output dimension per group
            K_blocks: Number of scale blocks along K dimension

        Returns:
            Time in seconds
        """
        # Input: (G, N, K_blocks) uint8 scales
        read_bytes = G * N * K_blocks * 1  # uint8

        # Output: Rearranged (G, N, K_blocks) float8 scales
        write_bytes = G * N * K_blocks * 1  # float8

        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9

        time_s = total_gb / self.memory_bandwidth_gbs
        return time_s


# =============================================================================
# Benchmark functions
# =============================================================================


def benchmark_cuda_function_in_microseconds(f, *args):
    """Benchmark a CUDA function and return time in microseconds"""
    return do_bench(lambda: f(*args), return_mode="median") * 1e3


def benchmark_torch_grouped_mm_fwd_bwd(x, w_t, offs, labels):
    """Benchmark torch._grouped_mm forward + backward"""
    x_clone = x.clone().requires_grad_(True)
    w_t_clone = w_t.clone().requires_grad_(True)

    fn = torch.compile(torch._grouped_mm, fullgraph=True)

    def wrapper():
        out = fn(x_clone, w_t_clone, offs=offs, out_dtype=torch.bfloat16)
        loss = torch.nn.functional.mse_loss(out, labels)
        loss.backward()

    time_ms = do_bench(wrapper, return_mode="median")
    return time_ms


def benchmark_mxfp8_grouped_mm_fwd_bwd(x, w_t, offs, labels, block_size=32):
    """Benchmark _to_mxfp8_then_scaled_grouped_mm forward + backward"""
    x_clone = x.clone().requires_grad_(True)
    w_t_clone = w_t.clone().requires_grad_(True)

    fn = torch.compile(_to_mxfp8_then_scaled_grouped_mm, fullgraph=True)

    # Set all parameters explicitly as variables for positional args
    A = x_clone
    B_t = w_t_clone
    offs_arg = offs
    block_size_arg = block_size
    out_dtype = torch.bfloat16
    emulated = False
    use_triton_for_dim0_cast = True
    scale_calculation_mode = MoEScaleCalculationMode.RCEIL

    def wrapper():
        out = fn(
            A,
            B_t,
            offs_arg,
            block_size_arg,
            out_dtype,
            emulated,
            use_triton_for_dim0_cast,
            scale_calculation_mode,
        )
        loss = torch.nn.functional.mse_loss(out, labels)
        loss.backward()

    time_ms = do_bench(wrapper, return_mode="median")
    return time_ms


def benchmark_triton_to_mxfp8_dim0(tensor, block_size=32):
    """Benchmark triton_to_mxfp8_dim0 kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: triton_to_mxfp8_dim0(tensor, inner_block_size=block_size)
    )


def benchmark_to_mxfp8_dim1_cuda(tensor, block_size=32):
    """Benchmark _to_mxfp8_dim1_kernel_wrapper with CUDA kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: _to_mxfp8_dim1_kernel_wrapper(
            tensor,
            block_size=block_size,
            elem_dtype=torch.float8_e4m3fn,
            hp_dtype=torch.bfloat16,
            gemm_kernel_choice=None,
            cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )
    )


def benchmark_to_mx(tensor, block_size=32):
    """Benchmark to_mx kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: _compiled_to_mx_wrapper(tensor, block_size, ScaleCalculationMode.RCEIL)
    )


def benchmark_mxfp8_quantize_cuda_3d(tensor, block_size=32):
    """Benchmark mxfp8_quantize_cuda_3d kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: mxfp8_quantize_cuda_3d(
            tensor, block_size=block_size, scaling_mode="rceil"
        )
    )


def benchmark_bf16_grouped_gemm(x_bf16, w_t_bf16, offs):
    """Benchmark BF16 grouped GEMM kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: torch._grouped_mm(x_bf16, w_t_bf16, offs=offs, out_dtype=torch.bfloat16)
    )


def benchmark_mxfp8_grouped_gemm(x_fp8, w_fp8, x_scales, w_scales, offs):
    """Benchmark MXFP8 grouped GEMM kernel"""
    return benchmark_cuda_function_in_microseconds(
        lambda: torch._scaled_grouped_mm(
            x_fp8, w_fp8, x_scales, w_scales, offs=offs, out_dtype=torch.bfloat16
        )
    )


# =============================================================================
# Helper functions
# =============================================================================


def generate_shape_configs(K, N, G):
    """Generate shape configurations varying only M dimension"""
    configs = []
    for M in [16384, 32768, 65536, 131072]:
        configs.append((M, K, N, G, f"M={M}"))
    return configs


# =============================================================================
# Main function
# =============================================================================


def run(
    K: int = 4096,
    N: int = 4096,
    G: int = 8,
    breakdown_M: int = None,
    outfile_speedup: str = "roofline_speedup_results.csv",
    outfile_quant_2d: str = "roofline_quant_2d_results.csv",
    outfile_quant_3d: str = "roofline_quant_3d_results.csv",
    plot_file: str = "roofline_unified.png",
    gpu_name: str = "NVIDIA B200",
    power_limit_percent: float = 100.0,
):
    """
    Generate unified roofline analysis for MXFP8 grouped GEMM.

    Args:
        K: Reduction dimension (default: 4096)
        N: Output dimension per group (default: 4096)
        G: Number of groups (default: 8)
        breakdown_M: M value to use for kernel breakdown analysis (default: None, uses largest M from configs)
        outfile_speedup: CSV file for speedup results
        outfile_quant_2d: CSV file for 2D quantization results
        outfile_quant_3d: CSV file for 3D quantization results
        plot_file: PNG file to save unified plot
        gpu_name: GPU model (default: B200)
        power_limit_percent: Power limit as percentage (0-100, default: 100.0)
    """
    print(f"GPU: {gpu_name}")
    print(f"Torch version: {torch.__version__}")
    print(f"\nFixed dimensions: K={K}, N={N}, G={G}")
    print(f"Power limit: {power_limit_percent}%")

    model = RooflineModel(gpu_name=gpu_name, power_limit_percent=power_limit_percent)

    print("\nGPU Specs:")
    print(f"  BF16 TFLOPS: {model.bf16_tflops}")
    print(f"  MXFP8 TFLOPS: {model.mxfp8_tflops}")
    print(f"  Memory Bandwidth: {model.memory_bandwidth_gbs} GB/s")

    configs = generate_shape_configs(K, N, G)

    # =============================================================================
    # 1. Net Speedup Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("NET SPEEDUP ANALYSIS (BF16 vs MXFP8)")
    print("=" * 80)

    speedup_results = []
    for M, K_val, N_val, G_val, desc in configs:
        result = model.compute_speedup(M, K_val, N_val, G_val)
        result_dict = {
            "M": M,
            "K": K_val,
            "N": N_val,
            "G": G_val,
            "description": desc,
            "bf16_roofline_time_ms": result["bf16_roofline_time_ms"],
            "mxfp8_roofline_quant_time_ms": result["mxfp8_roofline_quant_time_ms"],
            "mxfp8_roofline_gemm_time_ms": result["mxfp8_roofline_gemm_time_ms"],
            "mxfp8_roofline_total_time_ms": result["mxfp8_roofline_total_time_ms"],
            "roofline_speedup": result["roofline_speedup"],
            "roofline_quant_overhead_pct": (
                result["mxfp8_roofline_quant_time_ms"]
                / result["mxfp8_roofline_total_time_ms"]
            )
            * 100,
        }

        print(f"\nBenchmarking {desc}...")

        # Create test tensors
        x = torch.randn(M, K_val, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(G_val, N_val, K_val, dtype=torch.bfloat16, device="cuda")
        w_t = w.contiguous().transpose(-2, -1)
        offs = generate_jagged_offs(G_val, M)
        labels = torch.ones((M, N_val), device="cuda", dtype=torch.bfloat16)

        # Benchmark BF16
        bf16_actual_ms = benchmark_torch_grouped_mm_fwd_bwd(x, w_t, offs, labels)
        result_dict["bf16_actual_time_ms"] = bf16_actual_ms
        print(
            f"  BF16: Roofline={result['bf16_roofline_time_ms']:.3f}ms, Actual={bf16_actual_ms:.3f}ms"
        )

        # Benchmark MXFP8
        mxfp8_actual_ms = benchmark_mxfp8_grouped_mm_fwd_bwd(x, w_t, offs, labels)
        result_dict["mxfp8_actual_time_ms"] = mxfp8_actual_ms
        result_dict["actual_speedup"] = (
            bf16_actual_ms / mxfp8_actual_ms if bf16_actual_ms else None
        )
        print(
            f"  MXFP8: Roofline={result['mxfp8_roofline_total_time_ms']:.3f}ms, Actual={mxfp8_actual_ms:.3f}ms"
        )
        if result_dict["actual_speedup"]:
            print(f"  Actual Speedup: {result_dict['actual_speedup']:.3f}x")

        speedup_results.append(result_dict)

        # Clean up tensors to free GPU memory
        del x, w, w_t, offs, labels
        torch.cuda.empty_cache()

    df_speedup = pd.DataFrame(speedup_results)
    df_speedup.to_csv(outfile_speedup, index=False)
    print(f"\nSpeedup results saved to {outfile_speedup}")

    # =============================================================================
    # 2. 2D Quantization Kernel Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("2D QUANTIZATION KERNELS (Forward Pass)")
    print("=" * 80)

    quant_2d_results = []
    for M, K_val, _, _, desc in configs:
        roofline_time = model.compute_quant_2d_time(M, K_val)

        # Calculate bandwidth metrics
        read_bytes = M * K_val * 2
        write_bytes = M * K_val * 1 + M * (K_val // 32) * 1
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        roofline_bandwidth_gbs = model.memory_bandwidth_gbs

        result_dict = {
            "M": M,
            "K": K_val,
            "description": desc,
            "roofline_time_ms": roofline_time * 1000,
            "roofline_bandwidth_gbs": roofline_bandwidth_gbs,
            "total_gb": total_gb,
        }

        print(f"\nBenchmarking {desc}...")

        # Create test tensor
        tensor = torch.randn(M, K_val, dtype=torch.bfloat16, device="cuda")

        # Benchmark triton_to_mxfp8_dim0
        triton_dim0_time_us = benchmark_triton_to_mxfp8_dim0(tensor)
        triton_dim0_bandwidth_gbs = total_gb / (triton_dim0_time_us / 1e6)
        result_dict["triton_to_mxfp8_dim0_us"] = triton_dim0_time_us
        result_dict["triton_dim0_bandwidth_gbs"] = triton_dim0_bandwidth_gbs
        result_dict["triton_dim0_efficiency_pct"] = (
            triton_dim0_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  triton_to_mxfp8_dim0: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={triton_dim0_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['triton_dim0_efficiency_pct']:.1f}%"
        )

        # Benchmark triton_to_mxfp8_dim1 (CUDA)
        dim1_cuda_time_us = benchmark_to_mxfp8_dim1_cuda(tensor)
        dim1_cuda_bandwidth_gbs = total_gb / (dim1_cuda_time_us / 1e6)
        result_dict["to_mxfp8_dim1_cuda_us"] = dim1_cuda_time_us
        result_dict["dim1_cuda_bandwidth_gbs"] = dim1_cuda_bandwidth_gbs
        result_dict["cuda_dim1_efficiency_pct"] = (
            dim1_cuda_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  to_mxfp8_dim1_cuda: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={dim1_cuda_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cuda_dim1_efficiency_pct']:.1f}%"
        )

        # Benchmark to_mx
        # Warmup runs before benchmarking to ensure compiled function is optimized
        for _ in range(2):
            with torch.no_grad():
                _ = _compiled_to_mx_wrapper(tensor, 32, ScaleCalculationMode.RCEIL)

        to_mx_time_us = benchmark_to_mx(tensor)
        to_mx_bandwidth_gbs = total_gb / (to_mx_time_us / 1e6)
        result_dict["to_mx_us"] = to_mx_time_us
        result_dict["to_mx_bandwidth_gbs"] = to_mx_bandwidth_gbs
        result_dict["to_mx_efficiency_pct"] = (
            to_mx_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  to_mx: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={to_mx_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['to_mx_efficiency_pct']:.1f}%"
        )

        quant_2d_results.append(result_dict)

        # Clean up tensors to free GPU memory
        del tensor
        torch.cuda.empty_cache()

    df_quant_2d = pd.DataFrame(quant_2d_results)
    df_quant_2d.to_csv(outfile_quant_2d, index=False)
    print(f"\n2D quantization results saved to {outfile_quant_2d}")

    # =============================================================================
    # 3. 3D Quantization Kernel Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("3D QUANTIZATION KERNELS (Backward Pass - Weight Quantization)")
    print("=" * 80)

    quant_3d_results = []
    for M, K_val, N_val, G_val, desc in configs:
        roofline_time = model.compute_quant_3d_time(G_val, N_val, K_val)

        # Calculate bandwidth metrics
        read_bytes = G_val * N_val * K_val * 2
        write_bytes = G_val * N_val * K_val * 1 + G_val * N_val * (K_val // 32) * 1
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        roofline_bandwidth_gbs = model.memory_bandwidth_gbs

        result_dict = {
            "E": G_val,
            "N": N_val,
            "K": K_val,
            "description": desc,
            "roofline_time_ms": roofline_time * 1000,
            "roofline_bandwidth_gbs": roofline_bandwidth_gbs,
            "total_gb": total_gb,
        }

        print(f"\nBenchmarking {desc}...")

        # Create test tensor
        tensor = torch.randn(G_val, N_val, K_val, dtype=torch.bfloat16, device="cuda")

        # Benchmark mxfp8_quantize_cuda_3d
        cuda_3d_time_us = benchmark_mxfp8_quantize_cuda_3d(tensor)
        cuda_3d_bandwidth_gbs = total_gb / (cuda_3d_time_us / 1e6)
        result_dict["mxfp8_quantize_cuda_3d_us"] = cuda_3d_time_us
        result_dict["cuda_3d_bandwidth_gbs"] = cuda_3d_bandwidth_gbs
        result_dict["cuda_3d_efficiency_pct"] = (
            cuda_3d_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mxfp8_quantize_cuda_3d: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cuda_3d_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cuda_3d_efficiency_pct']:.1f}%"
        )

        quant_3d_results.append(result_dict)

        # Clean up tensors to free GPU memory
        del tensor
        torch.cuda.empty_cache()

    df_quant_3d = pd.DataFrame(quant_3d_results)
    df_quant_3d.to_csv(outfile_quant_3d, index=False)
    print(f"\n3D quantization results saved to {outfile_quant_3d}")

    # =============================================================================
    # 4. 2D Rearrange Kernels Analysis (Scale Blocking for Grouped GEMM)
    # =============================================================================
    print("\n" + "=" * 80)
    print("2D SCALE REARRANGE KERNELS (Scale Blocking for Grouped GEMM)")
    print("=" * 80)

    block_size = 32
    num_groups = G
    rearrange_results = []

    # M-groups configurations (forward pass input scales)
    for M, K_val, _, _, desc in configs:
        K_blocks = K_val // block_size

        # Calculate roofline time
        roofline_time = model.compute_rearrange_2d_M_groups_time(
            M,
            K_blocks,
        )

        # Calculate bandwidth metrics
        read_bytes = M * K_blocks * 1  # uint8
        write_bytes = M * K_blocks * 1  # float8
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        roofline_bandwidth_gbs = model.memory_bandwidth_gbs

        result_dict = {
            "kernel_type": "M_groups",
            "M": M,
            "K_dim": K_val,
            "K_blocks": K_blocks,
            "description": desc,
            "roofline_time_ms": roofline_time * 1000,
            "roofline_bandwidth_gbs": roofline_bandwidth_gbs,
            "total_gb": total_gb,
        }

        print(f"\nBenchmarking M-groups rearrange {desc}...")

        # Create test tensor (uint8 scales)
        input_tensor = torch.randint(
            low=0,
            high=256,
            size=(M, K_blocks),
            dtype=torch.uint8,
            device="cuda",
        )
        input_group_offsets = generate_jagged_offs(num_groups, M)

        # Benchmark triton kernel
        triton_out = triton_mx_block_rearrange_2d_M_groups(
            input_tensor, input_group_offsets
        )
        triton_time_us = benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_2d_M_groups,
            input_tensor,
            input_group_offsets,
        )
        triton_bandwidth_gbs = total_gb / (triton_time_us / 1e6)
        result_dict["triton_mx_block_rearrange_2d_M_groups_us"] = triton_time_us
        result_dict["triton_bandwidth_gbs"] = triton_bandwidth_gbs
        result_dict["triton_efficiency_pct"] = (
            triton_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  triton_mx_block_rearrange_2d_M_groups: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={triton_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['triton_efficiency_pct']:.1f}%"
        )

        rearrange_results.append(result_dict)

        # Clean up tensors
        del input_tensor, input_group_offsets, triton_out
        torch.cuda.empty_cache()

    # K-groups configurations (backward weight pass scales)
    for M, K_val, N_val, _, desc in configs:
        M_blocks = M // block_size

        # Calculate roofline time
        roofline_time = model.compute_rearrange_2d_K_groups_time(
            N_val,
            M_blocks,
        )

        # Calculate bandwidth metrics
        read_bytes = N_val * M_blocks * 1  # uint8
        write_bytes = N_val * M_blocks * 1  # float8
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        roofline_bandwidth_gbs = model.memory_bandwidth_gbs

        result_dict = {
            "kernel_type": "K_groups",
            "M": M,
            "N": N_val,
            "M_blocks": M_blocks,
            "description": desc,
            "roofline_time_ms": roofline_time * 1000,
            "roofline_bandwidth_gbs": roofline_bandwidth_gbs,
            "total_gb": total_gb,
        }

        print(f"\nBenchmarking K-groups rearrange {desc}...")

        # Create test tensor (uint8 scales from transposed quantization)
        input_tensor = torch.randint(
            low=0,
            high=256,
            size=(N_val, M_blocks),
            dtype=torch.uint8,
            device="cuda",
        )
        scale_group_offsets = generate_jagged_offs(num_groups, M) // block_size

        # Benchmark triton kernel
        triton_out = triton_mx_block_rearrange_2d_K_groups(
            input_tensor, scale_group_offsets
        )
        triton_time_us = benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_2d_K_groups,
            input_tensor,
            scale_group_offsets,
        )
        triton_bandwidth_gbs = total_gb / (triton_time_us / 1e6)
        result_dict["triton_mx_block_rearrange_2d_K_groups_us"] = triton_time_us
        result_dict["triton_bandwidth_gbs"] = triton_bandwidth_gbs
        result_dict["triton_efficiency_pct"] = (
            triton_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  triton_mx_block_rearrange_2d_K_groups: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={triton_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['triton_efficiency_pct']:.1f}%"
        )

        rearrange_results.append(result_dict)

        # Clean up tensors
        del input_tensor, scale_group_offsets, triton_out
        torch.cuda.empty_cache()

    df_rearrange = pd.DataFrame(rearrange_results)
    print("\n2D rearrange results completed")

    # =============================================================================
    # 4b. 3D Rearrange Kernels Analysis (Per-Group Scale Blocking)
    # =============================================================================
    print("\n" + "=" * 80)
    print("3D SCALE REARRANGE KERNELS (Per-Group Scale Blocking)")
    print("=" * 80)

    rearrange_3d_results = []
    for M, K_val, N_val, G_val, desc in configs:
        K_blocks = K_val // block_size

        # Calculate roofline time for 3D rearrangement
        # Input: (G, N, K_blocks) uint8 scales
        # Output: (G, N, K_blocks) float8 scales
        read_bytes = G_val * N_val * K_blocks * 1  # uint8
        write_bytes = G_val * N_val * K_blocks * 1  # float8
        total_bytes = read_bytes + write_bytes
        total_gb = total_bytes / 1e9
        roofline_time = total_gb / model.memory_bandwidth_gbs
        roofline_bandwidth_gbs = model.memory_bandwidth_gbs

        result_dict = {
            "M": M,
            "G": G_val,
            "N": N_val,
            "K": K_val,
            "K_blocks": K_blocks,
            "description": desc,
            "roofline_time_ms": roofline_time * 1000,
            "roofline_bandwidth_gbs": roofline_bandwidth_gbs,
            "total_gb": total_gb,
        }

        print(f"\nBenchmarking 3D rearrange {desc}...")

        # Create test tensor (uint8 scales)
        input_tensor = torch.randint(
            low=0,
            high=256,
            size=(G_val, N_val, K_blocks),
            dtype=torch.uint8,
            device="cuda",
        )

        # Benchmark triton kernel
        triton_out = triton_mx_block_rearrange_per_group_3d(input_tensor)
        triton_time_us = benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_per_group_3d,
            input_tensor,
        )
        triton_bandwidth_gbs = total_gb / (triton_time_us / 1e6)
        result_dict["triton_mx_block_rearrange_per_group_3d_us"] = triton_time_us
        result_dict["triton_bandwidth_gbs"] = triton_bandwidth_gbs
        result_dict["triton_efficiency_pct"] = (
            triton_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  triton_mx_block_rearrange_per_group_3d: "
            f"Roofline={roofline_bandwidth_gbs:.1f} GB/s, "
            f"Actual={triton_bandwidth_gbs:.1f} GB/s, "
            f"Efficiency={result_dict['triton_efficiency_pct']:.1f}%"
        )

        rearrange_3d_results.append(result_dict)

        # Clean up tensors
        del input_tensor, triton_out
        torch.cuda.empty_cache()

    df_rearrange_3d = pd.DataFrame(rearrange_3d_results)
    print("\n3D rearrange results completed")

    # =============================================================================
    # 5. Grouped GEMM Kernel Analysis
    # =============================================================================
    print("\n" + "=" * 80)
    print("GROUPED GEMM KERNEL ANALYSIS")
    print("=" * 80)

    grouped_gemm_results = []
    for M, K_val, N_val, G_val, desc in configs:
        # Calculate roofline compute time for 2D-3D GEMM
        roofline_gemm_time = model.compute_mxfp8_2d_3d_gemm_time(M, K_val, N_val)

        result_dict = {
            "M": M,
            "K": K_val,
            "N": N_val,
            "G": G_val,
            "description": desc,
            "roofline_gemm_time_ms": roofline_gemm_time * 1000,
            "roofline_tflops": model.mxfp8_tflops,
        }

        print(f"\nBenchmarking {desc}...")

        # Create test tensors
        x = torch.randn(M, K_val, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(G_val, N_val, K_val, dtype=torch.bfloat16, device="cuda")
        w_t = w.contiguous().transpose(-2, -1)
        offs = generate_jagged_offs(G_val, M)

        # Benchmark BF16 grouped GEMM
        bf16_gemm_time_us = benchmark_bf16_grouped_gemm(x, w_t, offs)

        # Calculate BF16 TFLOPS
        total_flops = 2 * M * K_val * N_val
        total_tflops = total_flops / 1e12
        bf16_actual_tflops = total_tflops / (bf16_gemm_time_us / 1e6)

        result_dict["bf16_gemm_time_ms"] = bf16_gemm_time_us / 1000
        result_dict["bf16_actual_tflops"] = bf16_actual_tflops
        result_dict["bf16_tflops_efficiency_pct"] = (
            bf16_actual_tflops / model.bf16_tflops
        ) * 100

        print(
            f"  BF16 Grouped GEMM: Roofline={model.bf16_tflops:.1f} TFLOPS, Actual={bf16_actual_tflops:.1f} TFLOPS, Efficiency={result_dict['bf16_tflops_efficiency_pct']:.1f}%"
        )

        # Convert to MXFP8 format
        x_scales, x_fp8 = to_mx(x, elem_dtype=torch.float8_e4m3fn, block_size=32)
        w_scales, w_fp8 = to_mx(
            w_t.transpose(-2, -1), elem_dtype=torch.float8_e4m3fn, block_size=32
        )

        # Convert scales to blocked format
        x_scales_blocked, _ = torch_to_blocked_2d_M_groups(
            x_scales, offs, block_size=32
        )
        w_scales_blocked = torch_to_blocked_per_group_3d(w_scales)

        # Benchmark the MXFP8 grouped GEMM kernel
        mxfp8_gemm_time_us = benchmark_mxfp8_grouped_gemm(
            x_fp8, w_fp8.transpose(-2, -1), x_scales_blocked, w_scales_blocked, offs
        )

        # Calculate MXFP8 actual TFLOPS
        mxfp8_actual_tflops = total_tflops / (mxfp8_gemm_time_us / 1e6)

        result_dict["mxfp8_gemm_time_ms"] = mxfp8_gemm_time_us / 1000
        result_dict["mxfp8_actual_tflops"] = mxfp8_actual_tflops
        result_dict["mxfp8_tflops_efficiency_pct"] = (
            mxfp8_actual_tflops / model.mxfp8_tflops
        ) * 100

        print(
            f"  MXFP8 Grouped GEMM: Roofline={model.mxfp8_tflops:.1f} TFLOPS, Actual={mxfp8_actual_tflops:.1f} TFLOPS, Efficiency={result_dict['mxfp8_tflops_efficiency_pct']:.1f}%"
        )

        # Calculate and print speedup
        gemm_speedup = mxfp8_actual_tflops / bf16_actual_tflops
        result_dict["gemm_speedup"] = gemm_speedup
        print(f"  GEMM Speedup (MXFP8 vs BF16): {gemm_speedup:.3f}x")

        grouped_gemm_results.append(result_dict)

        # Clean up tensors to free GPU memory
        del x, w, w_t, offs, x_fp8, x_scales, w_fp8, w_scales
        del x_scales_blocked, w_scales_blocked
        torch.cuda.empty_cache()

    df_grouped_gemm = pd.DataFrame(grouped_gemm_results)

    # =============================================================================
    # 6. 2D/2D Grouped GEMM Kernel Analysis (Backward Weight)
    # =============================================================================
    print("\n" + "=" * 80)
    print("2D/2D GROUPED GEMM KERNEL ANALYSIS (Backward Weight)")
    print("=" * 80)

    grouped_gemm_2d_2d_results = []
    for M, K_val, N_val, G_val, desc in configs:
        # For 2D/2D grouped GEMM: (N, M) @ (M, K) -> (N, K) per group
        # Total FLOPs = 2 * N * M * K (same as 2D/3D but different layout)
        result_dict = {
            "M": M,
            "K": K_val,
            "N": N_val,
            "G": G_val,
            "description": desc,
        }

        print(f"\nBenchmarking {desc}...")

        # Create test tensors for 2D/2D grouped GEMM
        # Simulate backward weight: grad_output.T @ input
        # We'll create grad_output and input, then quantize them
        grad_out = torch.randn(M, N_val, dtype=torch.bfloat16, device="cuda")
        x = torch.randn(M, K_val, dtype=torch.bfloat16, device="cuda")
        offs = generate_jagged_offs(G_val, M)

        # Benchmark BF16 2D/2D grouped GEMM
        # For BF16, we need grad_out_t = grad_out.t().contiguous() to get (N, M) row-major
        grad_out_t = grad_out.t().contiguous()
        bf16_gemm_2d_2d_time_us = benchmark_bf16_grouped_gemm(grad_out_t, x, offs)

        # Calculate BF16 TFLOPS
        total_flops = 2 * N_val * M * K_val
        total_tflops = total_flops / 1e12
        bf16_2d_2d_actual_tflops = total_tflops / (bf16_gemm_2d_2d_time_us / 1e6)

        result_dict["bf16_2d_2d_gemm_time_ms"] = bf16_gemm_2d_2d_time_us / 1000
        result_dict["bf16_2d_2d_actual_tflops"] = bf16_2d_2d_actual_tflops
        result_dict["bf16_2d_2d_tflops_efficiency_pct"] = (
            bf16_2d_2d_actual_tflops / model.bf16_tflops
        ) * 100

        print(
            f"  BF16 2D/2D Grouped GEMM: Roofline={model.bf16_tflops:.1f} TFLOPS, Actual={bf16_2d_2d_actual_tflops:.1f} TFLOPS, Efficiency={result_dict['bf16_2d_2d_tflops_efficiency_pct']:.1f}%"
        )

        # Convert to MXFP8 format for 2D/2D grouped GEMM
        # For 2D/2D, scales are computed along the K dimension (contracting dim)
        # Note: _to_mxfp8_dim1_kernel_wrapper returns the output TRANSPOSED

        # Quantize grad_out: (M, N) -> returns (N, M) transposed
        # This matches the pattern in scaled_grouped_mm.py backward pass line 410-420
        grad_out_mx = _to_mxfp8_dim1_kernel_wrapper(
            grad_out,
            32,
            elem_dtype=torch.float8_e4m3fn,
            hp_dtype=torch.bfloat16,
            gemm_kernel_choice=None,
            cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )
        grad_out_t_fp8 = grad_out_mx.qdata  # Shape: (N, M)
        grad_out_t_scales = grad_out_mx.scale  # Shape: (N, M//32)

        # Quantize x: (M, K) -> returns (K, M) transposed
        # This matches the pattern in scaled_grouped_mm.py backward pass line 426-436
        x_mx = _to_mxfp8_dim1_kernel_wrapper(
            x,
            32,
            elem_dtype=torch.float8_e4m3fn,
            hp_dtype=torch.bfloat16,
            gemm_kernel_choice=None,
            cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )
        x_t_fp8 = x_mx.qdata  # Shape: (K, M)
        x_t_scales = x_mx.scale  # Shape: (K, M//32)

        # Convert scales to blocked format for 2D/2D grouped mm
        scale_group_offsets = offs // 32
        grad_out_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
            grad_out_t_scales, scale_group_offsets
        )
        x_t_scales_blocked = triton_mx_block_rearrange_2d_K_groups(
            x_t_scales, scale_group_offsets
        )

        # Benchmark the MXFP8 2D/2D grouped GEMM kernel
        # Note: For 2D/2D grouped GEMM:
        # - Left operand (grad_out_t_fp8) should be row-major
        # - Right operand should be column-major (transpose without .contiguous())
        # Following the pattern in scaled_grouped_mm.py line 452: A_t_data.transpose(-2, -1)
        # x_t_fp8 has shape (K, M), transpose to (M, K) gives column-major layout
        mxfp8_gemm_2d_2d_time_us = benchmark_mxfp8_grouped_gemm(
            grad_out_t_fp8,
            x_t_fp8.transpose(-2, -1),
            grad_out_t_scales_blocked,
            x_t_scales_blocked,
            offs,
        )

        # Calculate MXFP8 actual TFLOPS
        mxfp8_2d_2d_actual_tflops = total_tflops / (mxfp8_gemm_2d_2d_time_us / 1e6)

        result_dict["mxfp8_2d_2d_gemm_time_ms"] = mxfp8_gemm_2d_2d_time_us / 1000
        result_dict["mxfp8_2d_2d_actual_tflops"] = mxfp8_2d_2d_actual_tflops
        result_dict["mxfp8_2d_2d_tflops_efficiency_pct"] = (
            mxfp8_2d_2d_actual_tflops / model.mxfp8_tflops
        ) * 100

        print(
            f"  MXFP8 2D/2D Grouped GEMM: Roofline={model.mxfp8_tflops:.1f} TFLOPS, Actual={mxfp8_2d_2d_actual_tflops:.1f} TFLOPS, Efficiency={result_dict['mxfp8_2d_2d_tflops_efficiency_pct']:.1f}%"
        )

        # Calculate and print speedup
        gemm_2d_2d_speedup = mxfp8_2d_2d_actual_tflops / bf16_2d_2d_actual_tflops
        result_dict["gemm_2d_2d_speedup"] = gemm_2d_2d_speedup
        print(f"  GEMM Speedup (MXFP8 vs BF16): {gemm_2d_2d_speedup:.3f}x")

        grouped_gemm_2d_2d_results.append(result_dict)

        # Clean up tensors to free GPU memory
        del grad_out, grad_out_t, x, offs
        del grad_out_t_fp8, grad_out_t_scales, x_t_fp8, x_t_scales
        del grad_out_t_scales_blocked, x_t_scales_blocked
        torch.cuda.empty_cache()

    df_grouped_gemm_2d_2d = pd.DataFrame(grouped_gemm_2d_2d_results)

    # =============================================================================
    # 7. Generate Unified Plots
    # =============================================================================
    print("\n" + "=" * 80)
    print("GENERATING UNIFIED PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(24, 12))

    # Plot 1: Net Speedup
    ax1 = axes[0, 0]
    ax1.plot(
        df_speedup["M"],
        df_speedup["roofline_speedup"],
        marker="o",
        linewidth=2,
        linestyle=":",
        label="Roofline Model",
    )
    ax1.plot(
        df_speedup["M"],
        df_speedup["actual_speedup"],
        marker="s",
        linewidth=2,
        linestyle="-",
        label="Actual Implementation",
        color="purple",
    )
    ax1.axhline(
        y=1.0,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label="1x Baseline (No Speedup)",
    )
    ax1.set_xlabel("Local Batch Size x Sequence Length (M)", fontsize=12)
    ax1.set_ylabel("Speedup (MXFP8 vs BF16)", fontsize=12)
    ax1.set_title(f"Net Speedup vs Batch Size (K={K}, N={N}, G={G})", fontsize=13)
    ax1.set_ylim(0, 2)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df_speedup["M"])
    ax1.set_xticklabels([f"{int(m):,}" for m in df_speedup["M"]])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: 2D Quantization + Rearrange Kernels (Bandwidth %)
    ax2 = axes[0, 1]
    # 2D Quantization kernels
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["triton_dim0_efficiency_pct"],
        marker="s",
        linewidth=2,
        linestyle="-",
        label="triton_to_mxfp8_dim0",
        color="blue",
    )
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["cuda_dim1_efficiency_pct"],
        marker="d",
        linewidth=2,
        linestyle="-",
        label="to_mxfp8_dim1_cuda",
        color="orange",
    )
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["to_mx_efficiency_pct"],
        marker="^",
        linewidth=2,
        linestyle="-",
        label="to_mx",
        color="green",
    )
    # 2D Rearrange kernels
    df_m_groups = df_rearrange[df_rearrange["kernel_type"] == "M_groups"]
    df_k_groups = df_rearrange[df_rearrange["kernel_type"] == "K_groups"]
    ax2.plot(
        df_m_groups["M"],
        df_m_groups["triton_efficiency_pct"],
        marker="^",
        linewidth=2,
        linestyle="--",
        label="triton M-groups scale blocked format",
        color="purple",
    )
    ax2.plot(
        df_k_groups["M"],
        df_k_groups["triton_efficiency_pct"],
        marker="d",
        linewidth=2,
        linestyle="--",
        label="triton K-groups scale blocked format",
        color="red",
    )
    ax2.set_xlabel("Local Batch Size x Sequence Length (M)", fontsize=12)
    ax2.set_ylabel("Bandwidth Utilization (% of Peak)", fontsize=12)
    ax2.set_title(f"2D Quantization + Block Format Kernels (K={K}, N={N})", fontsize=13)
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(df_quant_2d["M"])
    ax2.set_xticklabels([f"{int(m):,}" for m in df_quant_2d["M"]])
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Grouped GEMM Kernel Speedup (MXFP8 vs BF16)
    ax3 = axes[1, 0]
    # Calculate speedup for 2D/3D grouped GEMM
    speedup_2d_3d = (
        df_grouped_gemm["mxfp8_actual_tflops"] / df_grouped_gemm["bf16_actual_tflops"]
    )
    # Calculate speedup for 2D/2D grouped GEMM
    speedup_2d_2d = (
        df_grouped_gemm_2d_2d["mxfp8_2d_2d_actual_tflops"]
        / df_grouped_gemm_2d_2d["bf16_2d_2d_actual_tflops"]
    )

    ax3.plot(
        df_grouped_gemm["M"],
        speedup_2d_3d,
        marker="s",
        linewidth=2,
        linestyle="-",
        label="2D/3D GEMM (fwd/bwd input)",
        color="purple",
    )
    ax3.plot(
        df_grouped_gemm_2d_2d["M"],
        speedup_2d_2d,
        marker="d",
        linewidth=2,
        linestyle="--",
        label="2D/2D GEMM (bwd weight)",
        color="orange",
    )
    ax3.axhline(
        y=1.0,
        color="red",
        linestyle=":",
        linewidth=1.5,
        label="1x (No Speedup)",
    )
    ax3.set_xlabel("Local Batch Size x Sequence Length (M)", fontsize=12)
    ax3.set_ylabel("Speedup (MXFP8 vs BF16)", fontsize=12)
    ax3.set_title(
        f"Grouped GEMM Kernel Speedup: MXFP8 over BF16 (K={K}, N={N}, G={G})",
        fontsize=13,
    )
    ax3.set_xscale("log", base=2)
    ax3.set_xticks(df_grouped_gemm["M"])
    ax3.set_xticklabels([f"{int(m):,}" for m in df_grouped_gemm["M"]])
    # Calculate y-axis limits to ensure all data points are visible
    all_speedups = pd.concat([speedup_2d_3d, speedup_2d_2d])
    max_speedup = all_speedups.max()
    min_speedup = all_speedups.min()
    y_margin = (max_speedup - min_speedup) * 0.1  # 10% margin
    ax3.set_ylim(max(0, min_speedup - y_margin), max_speedup + y_margin)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Empty placeholder (reserved for future use)
    ax4 = axes[0, 2]
    ax4.text(
        0.5,
        0.5,
        "Reserved for Future Analysis",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax4.transAxes,
        fontsize=14,
        color="gray",
    )
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.grid(False)

    # Plot 5: 3D Quantization + Rearrange Kernels (Bandwidth %)
    ax5 = axes[1, 1]
    m_values = [int(desc.split("M=")[1]) for desc in df_quant_3d["description"]]
    # 3D Quantization kernel
    ax5.plot(
        m_values,
        df_quant_3d["cuda_3d_efficiency_pct"],
        marker="s",
        linewidth=2,
        linestyle="-",
        label="mxfp8_quantize_cuda_3d",
        color="red",
    )
    # 3D Rearrange kernel
    ax5.plot(
        df_rearrange_3d["M"],
        df_rearrange_3d["triton_efficiency_pct"],
        marker="^",
        linewidth=2,
        linestyle="--",
        label="triton per-group 3D scale blocked format",
        color="purple",
    )
    ax5.set_xlabel("Local Batch Size x Sequence Length (M)", fontsize=12)
    ax5.set_ylabel("Bandwidth Utilization (% of Peak)", fontsize=12)
    ax5.set_title(
        f"3D Quantization + Block Format Kernels (E={G}, N={N}, K={K})", fontsize=13
    )
    ax5.set_xscale("log", base=2)
    ax5.set_xticks(m_values)
    ax5.set_xticklabels([f"{int(m):,}" for m in m_values])
    ax5.set_ylim(0, 100)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Kernel Breakdown Stacked Bar Chart
    ax6 = axes[1, 2]

    # Use configurable M value for detailed kernel breakdown
    if breakdown_M is None:
        M_large = configs[-1][0]  # Default: Last config has largest M
    else:
        M_large = breakdown_M
        # Validate that breakdown_M exists in configs
        if M_large not in [config[0] for config in configs]:
            print(
                f"\nWarning: breakdown_M={M_large} not in benchmark configs. Using default M={configs[-1][0]}"
            )
            M_large = configs[-1][0]

    K_val = K
    N_val = N
    G_val = G
    block_size = 32

    # Extract actual measured times from benchmark results (in ms)
    # Get the row corresponding to M_large
    idx_large = df_quant_2d[df_quant_2d["M"] == M_large].index[0]

    # Forward pass kernel times (actual measurements in ms)
    # Input quantization: use triton_to_mxfp8_dim0 for (M, K)
    fwd_input_quant_ms = df_quant_2d.loc[idx_large, "triton_to_mxfp8_dim0_us"] / 1000

    # Weight quantization: use mxfp8_quantize_cuda_3d for (G, N, K)
    idx_3d_large = df_quant_3d[df_quant_3d["description"] == f"M={M_large}"].index[0]
    fwd_weight_quant_ms = (
        df_quant_3d.loc[idx_3d_large, "mxfp8_quantize_cuda_3d_us"] / 1000
    )

    # Input scale rearrangement: M-groups for (M, K//32)
    idx_m_groups = df_rearrange[
        (df_rearrange["kernel_type"] == "M_groups") & (df_rearrange["M"] == M_large)
    ].index[0]
    fwd_input_scale_rearrange_ms = (
        df_rearrange.loc[idx_m_groups, "triton_mx_block_rearrange_2d_M_groups_us"]
        / 1000
    )

    # Weight scale rearrangement: 3D per-group for (G, N, K//32)
    idx_3d_rearrange = df_rearrange_3d[df_rearrange_3d["M"] == M_large].index[0]
    fwd_weight_scale_rearrange_ms = (
        df_rearrange_3d.loc[
            idx_3d_rearrange, "triton_mx_block_rearrange_per_group_3d_us"
        ]
        / 1000
    )

    # GEMM: use actual MXFP8 2D/3D grouped GEMM time
    idx_gemm = df_grouped_gemm[df_grouped_gemm["M"] == M_large].index[0]
    fwd_gemm_ms = df_grouped_gemm.loc[idx_gemm, "mxfp8_gemm_time_ms"]

    # Backward input pass - need to run additional benchmarks for (M, N) quantization
    # For grad_output quantization (M, N), we can estimate using the 2D kernel with N instead of K
    # Create and benchmark the tensors
    print(f"\nRunning additional benchmarks for kernel breakdown (M={M_large})...")

    # Backward input: grad_output quantization (M, N)
    grad_out_tensor = torch.randn(M_large, N_val, dtype=torch.bfloat16, device="cuda")
    bwd_input_grad_quant_ms = benchmark_triton_to_mxfp8_dim0(grad_out_tensor) / 1000

    # Backward input: weight quantization is same as forward (reuse)
    bwd_input_weight_quant_ms = fwd_weight_quant_ms

    # Backward input: grad scale rearrangement for (M, N//32) - M-groups
    grad_scales = torch.randint(
        0, 256, size=(M_large, N_val // block_size), dtype=torch.uint8, device="cuda"
    )
    grad_offs = generate_jagged_offs(G_val, M_large)
    bwd_input_grad_scale_rearrange_ms = (
        benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_2d_M_groups, grad_scales, grad_offs
        )
        / 1000
    )

    # Backward input: weight scale rearrangement is same as forward (reuse)
    bwd_input_weight_scale_rearrange_ms = fwd_weight_scale_rearrange_ms

    # Backward input: GEMM (same shape as forward, reuse)
    bwd_input_gemm_ms = fwd_gemm_ms

    # Backward weight pass
    # Grad.T quantization (N, M)
    grad_out_t_tensor = torch.randn(N_val, M_large, dtype=torch.bfloat16, device="cuda")
    bwd_weight_grad_quant_ms = benchmark_triton_to_mxfp8_dim0(grad_out_t_tensor) / 1000

    # Input quantization (M, K) - same as forward
    bwd_weight_input_quant_ms = fwd_input_quant_ms

    # Grad.T scale rearrangement (N, M//32) - K-groups
    idx_k_groups = df_rearrange[
        (df_rearrange["kernel_type"] == "K_groups") & (df_rearrange["M"] == M_large)
    ].index[0]
    bwd_weight_grad_scale_rearrange_ms = (
        df_rearrange.loc[idx_k_groups, "triton_mx_block_rearrange_2d_K_groups_us"]
        / 1000
    )

    # Input scale rearrangement - need K-groups for (K, M//32)
    input_scales_k = torch.randint(
        0, 256, size=(K_val, M_large // block_size), dtype=torch.uint8, device="cuda"
    )
    scale_group_offs = generate_jagged_offs(G_val, M_large) // block_size
    bwd_weight_input_scale_rearrange_ms = (
        benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_2d_K_groups, input_scales_k, scale_group_offs
        )
        / 1000
    )

    # GEMM: use actual MXFP8 2D/2D grouped GEMM time
    idx_gemm_2d2d = df_grouped_gemm_2d_2d[df_grouped_gemm_2d_2d["M"] == M_large].index[
        0
    ]
    bwd_weight_gemm_ms = df_grouped_gemm_2d_2d.loc[
        idx_gemm_2d2d, "mxfp8_2d_2d_gemm_time_ms"
    ]

    # Clean up temporary tensors
    del (
        grad_out_tensor,
        grad_scales,
        grad_offs,
        grad_out_t_tensor,
        input_scales_k,
        scale_group_offs,
    )
    torch.cuda.empty_cache()

    # Data for stacked bars
    passes = ["Forward", "Backward\nInput", "Backward\nWeight"]

    # Stack components (bottom to top)
    quant_1 = [fwd_input_quant_ms, bwd_input_grad_quant_ms, bwd_weight_grad_quant_ms]
    quant_2 = [
        fwd_weight_quant_ms,
        bwd_input_weight_quant_ms,
        bwd_weight_input_quant_ms,
    ]
    rearrange_1 = [
        fwd_input_scale_rearrange_ms,
        bwd_input_grad_scale_rearrange_ms,
        bwd_weight_grad_scale_rearrange_ms,
    ]
    rearrange_2 = [
        fwd_weight_scale_rearrange_ms,
        bwd_input_weight_scale_rearrange_ms,
        bwd_weight_input_scale_rearrange_ms,
    ]
    gemm = [fwd_gemm_ms, bwd_input_gemm_ms, bwd_weight_gemm_ms]

    # Calculate cumulative bottoms for stacking
    bottom_quant_2 = quant_1
    bottom_rearrange_1 = [quant_1[i] + quant_2[i] for i in range(3)]
    bottom_rearrange_2 = [bottom_rearrange_1[i] + rearrange_1[i] for i in range(3)]
    bottom_gemm = [bottom_rearrange_2[i] + rearrange_2[i] for i in range(3)]

    # Create stacked bars
    x_pos = range(len(passes))
    bar_width = 0.6

    ax6.bar(x_pos, quant_1, bar_width, label="Input/Grad Quant", color="#1f77b4")
    ax6.bar(
        x_pos,
        quant_2,
        bar_width,
        bottom=bottom_quant_2,
        label="Weight Quant",
        color="#ff7f0e",
    )
    ax6.bar(
        x_pos,
        rearrange_1,
        bar_width,
        bottom=bottom_rearrange_1,
        label="Input/Grad Rearrange",
        color="#2ca02c",
    )
    ax6.bar(
        x_pos,
        rearrange_2,
        bar_width,
        bottom=bottom_rearrange_2,
        label="Weight Rearrange",
        color="#d62728",
    )
    ax6.bar(x_pos, gemm, bar_width, bottom=bottom_gemm, label="GEMM", color="#9467bd")

    # Formatting
    ax6.set_ylabel("Time (ms)", fontsize=12)
    ax6.set_title(f"Kernel Breakdown (M={M_large:,})", fontsize=13)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(passes, fontsize=10)
    ax6.grid(True, alpha=0.3, axis="y")

    # Add total time labels on top of each bar
    totals = [
        sum([quant_1[i], quant_2[i], rearrange_1[i], rearrange_2[i], gemm[i]])
        for i in range(3)
    ]
    for i, (pos, total) in enumerate(zip(x_pos, totals)):
        ax6.text(pos, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9)

    # Add BF16 GEMM baseline reference lines
    # Forward and Backward Input use 2D/3D GEMM
    bf16_fwd_gemm_ms = df_grouped_gemm.loc[idx_gemm, "bf16_gemm_time_ms"]
    # Backward Weight uses 2D/2D GEMM
    bf16_bwd_weight_gemm_ms = df_grouped_gemm_2d_2d.loc[
        idx_gemm_2d2d, "bf16_2d_2d_gemm_time_ms"
    ]

    # Draw horizontal lines for BF16 baseline at each bar position
    bar_width_visual = 0.4  # Visual width for the line
    ax6.plot(
        [x_pos[0] - bar_width_visual, x_pos[0] + bar_width_visual],
        [bf16_fwd_gemm_ms, bf16_fwd_gemm_ms],
        color="red",
        linestyle="--",
        linewidth=2,
        label="BF16 GEMM baseline",
    )
    ax6.plot(
        [x_pos[1] - bar_width_visual, x_pos[1] + bar_width_visual],
        [bf16_fwd_gemm_ms, bf16_fwd_gemm_ms],
        color="red",
        linestyle="--",
        linewidth=2,
    )
    ax6.plot(
        [x_pos[2] - bar_width_visual, x_pos[2] + bar_width_visual],
        [bf16_bwd_weight_gemm_ms, bf16_bwd_weight_gemm_ms],
        color="red",
        linestyle="--",
        linewidth=2,
    )

    # Add legend after all plot elements are added
    ax6.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    print(f"\nUnified plot saved to {plot_file}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(
        f"""
Net Speedup Analysis:
  Average roofline speedup: {df_speedup["roofline_speedup"].mean():.3f}x
  Average actual speedup: {df_speedup["actual_speedup"].mean():.3f}x
  Median actual speedup: {df_speedup["actual_speedup"].median():.3f}x

2D Quantization Kernels:
  triton_to_mxfp8_dim0 avg efficiency: {df_quant_2d["triton_dim0_efficiency_pct"].mean():.1f}%
  to_mxfp8_dim1_cuda avg efficiency: {df_quant_2d["cuda_dim1_efficiency_pct"].mean():.1f}%
  to_mx avg efficiency: {df_quant_2d["to_mx_efficiency_pct"].mean():.1f}%

3D Quantization Kernels:
  mxfp8_quantize_cuda_3d avg efficiency: {df_quant_3d["cuda_3d_efficiency_pct"].mean():.1f}%

Grouped GEMM Kernel:
  torch._grouped_mm (BF16 2D/3D) avg efficiency: {df_grouped_gemm["bf16_tflops_efficiency_pct"].mean():.1f}%
  torch._scaled_grouped_mm (MXFP8 2D/3D) avg efficiency: {df_grouped_gemm["mxfp8_tflops_efficiency_pct"].mean():.1f}%
  torch._grouped_mm (BF16 2D/2D) avg efficiency: {df_grouped_gemm_2d_2d["bf16_2d_2d_tflops_efficiency_pct"].mean():.1f}%
  torch._scaled_grouped_mm (MXFP8 2D/2D) avg efficiency: {df_grouped_gemm_2d_2d["mxfp8_2d_2d_tflops_efficiency_pct"].mean():.1f}%

Configuration:
  K={K}, N={N}, G={G}
  Power Limit: {power_limit_percent}%
  Peak BW: {model.memory_bandwidth_gbs:.1f} GB/s
  Peak BF16 TFLOPS: {model.bf16_tflops:.1f} TFLOPS
  Peak MXFP8 TFLOPS: {model.mxfp8_tflops:.1f} TFLOPS
"""
    )
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(run)

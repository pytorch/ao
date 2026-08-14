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
    _mx_block_rearrange_2d_k_groups_cutedsl,
    mx_block_rearrange_2d_M_groups_cuda,
    mxfp8_quantize_2d_1x32_cutedsl,
    mxfp8_quantize_2d_32x1_cutedsl,
    torch_to_blocked_2d_M_groups,
    triton_mx_block_rearrange_2d_K_groups,
    triton_mx_block_rearrange_per_group_3d,
)
from torchao.prototype.moe_training.kernels.mxfp8.cutedsl_rearrange_2d_m_groups import (
    _mx_block_rearrange_2d_m_groups_cutedsl,
)
from torchao.prototype.moe_training.kernels.mxfp8.quant import (
    mxfp8_quantize_cuda_3d,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    ScaleCalculationMode as MoEScaleCalculationMode,
)
from torchao.prototype.moe_training.mxfp8_grouped_mm import (
    _to_mxfp8_then_scaled_grouped_mm,
)
from torchao.prototype.moe_training.utils import generate_jagged_offs
from torchao.prototype.mx_formats.config import (
    MXFP8Dim0CastKernelChoice,
    MXFP8Dim1CastKernelChoice,
    ScaleCalculationMode,
)
from torchao.prototype.mx_formats.kernels import triton_to_mxfp8_dim0
from torchao.prototype.mx_formats.utils import _to_mxfp8_dim1_kernel_wrapper
from torchao.quantization.quantize_.common import KernelPreference
from torchao.testing.training.roofline_utils import (
    calibrate_specs,
    gpu_name_to_specs,
)


class RooflineModel:
    """Roofline model for grouped GEMM on B200 GPU"""

    def __init__(
        self,
        gpu_name="NVIDIA B200",
        power_limit_percent=100.0,
        gpu_specs=None,
    ):
        """
        Args:
            gpu_name: GPU model name
            power_limit_percent: Power limit as percentage (0-100). Default 100.0
        """
        power_multiplier = power_limit_percent / 100.0

        if gpu_specs is not None:
            self.gpu_specs = gpu_specs
        elif gpu_name in gpu_name_to_specs:
            self.gpu_specs = gpu_name_to_specs[gpu_name]
        else:
            raise ValueError(f"Unsupported GPU: {gpu_name}")

        if self.gpu_specs is not None:
            bf16_gemm_pct = self.gpu_specs.get(
                "pct_achievable_bf16_gemm_tops",
                self.gpu_specs["pct_achievable_gemm_tops"],
            )
            fp8_gemm_pct = self.gpu_specs.get(
                "pct_achievable_fp8_gemm_tops",
                self.gpu_specs["pct_achievable_gemm_tops"],
            )
            self.bf16_tflops = (
                (self.gpu_specs["bf16_peak_tops"] / 1e12)
                * bf16_gemm_pct
                * power_multiplier
            )
            self.mxfp8_tflops = (
                (self.gpu_specs["fp8_peak_tops"] / 1e12)
                * fp8_gemm_pct
                * power_multiplier
            )
            self.memory_bandwidth_gbs = (
                (self.gpu_specs["peak_mem_bw_bytes_sec"] / 1e9)
                * self.gpu_specs["pct_achievable_mem_bw"]
                * power_multiplier
            )

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
        """Compute time for MXFP8 forward + backward pass."""
        block_size = 32

        # Forward: (M, K) @ (G, K, N)^T -> (M, N) [2D-3D]
        fwd_quant_time = self.compute_mxfp8_fwd_quant_time(M, K, G, N)
        # Forward scale rearrangement:
        # - Input scales (M, K//32) -> M-groups rearrangement
        # - Weight scales are emitted directly in blocked layout by the 3D kernel
        fwd_input_scale_rearrange_time = self.compute_rearrange_2d_M_groups_time(
            M, K // block_size
        )
        fwd_gemm_time = self.compute_mxfp8_2d_3d_gemm_time(M, K, N)

        # Backward input: (M, N) @ (G, N, K) -> (M, K) [2D-3D]
        bwd_input_quant_time = self.compute_mxfp8_bwd_input_quant_time(M, K, G, N)
        # Backward input scale rearrangement:
        # - grad_output scales (M, N//32) -> M-groups rearrangement
        # - Weight scales are emitted directly in blocked layout by the 3D kernel
        bwd_input_grad_scale_rearrange_time = self.compute_rearrange_2d_M_groups_time(
            M, N // block_size
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
            + fwd_gemm_time
            + bwd_input_quant_time
            + bwd_input_grad_scale_rearrange_time
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


# The cast kernels each backend label selects.
_BACKEND_CAST_KERNEL_CHOICES = {
    "cutedsl": (
        MXFP8Dim0CastKernelChoice.CUTEDSL,
        MXFP8Dim1CastKernelChoice.CUTEDSL,
    ),
    "legacy": (
        MXFP8Dim0CastKernelChoice.TRITON,
        MXFP8Dim1CastKernelChoice.CUDA,
    ),
}


def benchmark_mxfp8_grouped_mm_fwd_bwd(x, w_t, offs, labels, backend: str):
    """Benchmark _to_mxfp8_then_scaled_grouped_mm forward + backward"""
    dim0_choice, dim1_choice = _BACKEND_CAST_KERNEL_CHOICES[backend]
    torch._dynamo.reset()
    x_clone = x.clone().requires_grad_(True)
    w_t_clone = w_t.clone().requires_grad_(True)

    fn = torch.compile(_to_mxfp8_then_scaled_grouped_mm, fullgraph=True)

    def wrapper():
        out = fn(
            x_clone,
            w_t_clone,
            offs=offs,
            out_dtype=torch.bfloat16,
            kernel_preference=KernelPreference.AUTO,
            wgrad_with_hp=False,
            scale_calculation_mode=MoEScaleCalculationMode.RCEIL,
            pad_token_groups_for_grouped_mm=False,
            mxfp8_dim0_cast_kernel_choice=dim0_choice,
            mxfp8_dim1_cast_kernel_choice=dim1_choice,
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
            kernel_preference=None,
            cast_kernel_choice=MXFP8Dim1CastKernelChoice.CUDA,
            scale_calculation_mode=ScaleCalculationMode.RCEIL,
        )
    )


def benchmark_mxfp8_quantize_2d_1x32_cutedsl(tensor):
    return benchmark_cuda_function_in_microseconds(
        lambda: mxfp8_quantize_2d_1x32_cutedsl(tensor)
    )


def benchmark_mxfp8_quantize_2d_32x1_cutedsl(tensor, block_size=32):
    return benchmark_cuda_function_in_microseconds(
        lambda: mxfp8_quantize_2d_32x1_cutedsl(
            tensor,
            block_size=block_size,
            blocked_scale_output=False,
        )
    )


def benchmark_mxfp8_quantize_cuda_3d(tensor, block_size=32):
    """Benchmark the 3D 32x1 quantizer on its input tensor."""
    return benchmark_cuda_function_in_microseconds(
        lambda: mxfp8_quantize_cuda_3d(
            tensor,
            block_size=block_size,
            scale_block_dim1=block_size,
            scale_block_dim2=1,
            scaling_mode="rceil",
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
    backend: str = "cutedsl",
    outfile_speedup: str = "roofline_speedup_results.csv",
    outfile_quant_2d: str = "roofline_quant_2d_results.csv",
    outfile_quant_3d: str = "roofline_quant_3d_results.csv",
    plot_file: str = "roofline_unified.png",
    gpu_name: str = "NVIDIA B200",
    power_limit_percent: float = 100.0,
    calibrate_roofline: bool = False,
    calibration_m: int = 16384,
    calibration_n: int = 8192,
    calibration_k: int = 8192,
    calibration_copy_numel: int = 256 * 1024 * 1024,
):
    """
    Generate unified roofline analysis for MXFP8 grouped GEMM.

    Args:
        K: Reduction dimension (default: 4096)
        N: Output dimension per group (default: 4096)
        G: Number of groups (default: 8)
        breakdown_M: M value to use for kernel breakdown analysis (default: None, uses largest M from configs)
        backend: MXFP8 grouped-mm backend to benchmark: legacy, cutedsl, or both
        outfile_speedup: CSV file for speedup results
        outfile_quant_2d: CSV file for 2D quantization results
        outfile_quant_3d: CSV file for 3D quantization results
        plot_file: PNG file to save unified plot
        gpu_name: GPU model (default: B200)
        power_limit_percent: Power limit as percentage (0-100, default: 100.0)
        calibrate_roofline: Measure achievable GEMM and memory ceilings on this GPU
        calibration_m: M dimension for GEMM calibration
        calibration_n: N dimension for GEMM calibration
        calibration_k: K dimension for GEMM calibration
        calibration_copy_numel: Number of bf16 elements for memory calibration
    """
    K = 4096 if K == "" else int(K)
    N = 4096 if N == "" else int(N)
    G = 8 if G == "" else int(G)
    breakdown_M = None if breakdown_M in (None, "") else int(breakdown_M)
    power_limit_percent = (
        100.0 if power_limit_percent == "" else float(power_limit_percent)
    )
    if isinstance(calibrate_roofline, str):
        calibrate_roofline = calibrate_roofline.lower() in ("1", "true", "yes")
    calibration_m = int(calibration_m)
    calibration_n = int(calibration_n)
    calibration_k = int(calibration_k)
    calibration_copy_numel = int(calibration_copy_numel)

    print(f"GPU: {gpu_name}")
    print(f"Torch version: {torch.__version__}")
    print(f"\nFixed dimensions: K={K}, N={N}, G={G}")
    assert backend in ("legacy", "cutedsl", "both")
    mxfp8_backends = ("legacy", "cutedsl") if backend == "both" else (backend,)
    print(f"MXFP8 backend: {backend}")
    print(f"Power limit: {power_limit_percent}%")

    gpu_specs = None
    if calibrate_roofline:
        print("\nCalibrating roofline on current GPU...")
        gpu_specs = calibrate_specs(
            gpu_name=gpu_name,
            mm_shape=(calibration_m, calibration_n, calibration_k),
            copy_numel=calibration_copy_numel,
        )
        print(
            "  copy bandwidth: "
            f"{gpu_specs['calibrated_copy_bandwidth_bytes_sec'] / 1e9:.1f} GB/s"
        )
        print(f"  BF16 GEMM: {gpu_specs['calibrated_bf16_tops'] / 1e12:.1f} TFLOP/s")
        print(f"  FP8 GEMM: {gpu_specs['calibrated_fp8_tops'] / 1e12:.1f} TFLOP/s")
        print(
            "  measured fractions: "
            f"mem={gpu_specs['pct_achievable_mem_bw']:.3f}, "
            f"bf16={gpu_specs['pct_achievable_bf16_gemm_tops']:.3f}, "
            f"fp8={gpu_specs['pct_achievable_fp8_gemm_tops']:.3f}"
        )

    model = RooflineModel(
        gpu_name=gpu_name,
        power_limit_percent=power_limit_percent,
        gpu_specs=gpu_specs,
    )

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
        offs = generate_jagged_offs(G_val, M, multiple_of=128)
        labels = torch.ones((M, N_val), device="cuda", dtype=torch.bfloat16)

        # Benchmark BF16
        bf16_actual_ms = benchmark_torch_grouped_mm_fwd_bwd(x, w_t, offs, labels)
        result_dict["bf16_actual_time_ms"] = bf16_actual_ms
        print(
            f"  BF16: Roofline={result['bf16_roofline_time_ms']:.3f}ms, Actual={bf16_actual_ms:.3f}ms"
        )

        for mxfp8_backend in mxfp8_backends:
            mxfp8_actual_ms = benchmark_mxfp8_grouped_mm_fwd_bwd(
                x, w_t, offs, labels, mxfp8_backend
            )
            backend_time_key = f"mxfp8_actual_time_ms_{mxfp8_backend}"
            backend_speedup_key = f"actual_speedup_{mxfp8_backend}"
            result_dict[backend_time_key] = mxfp8_actual_ms
            result_dict[backend_speedup_key] = (
                bf16_actual_ms / mxfp8_actual_ms if bf16_actual_ms else None
            )
            if backend != "both":
                result_dict["mxfp8_actual_time_ms"] = mxfp8_actual_ms
                result_dict["actual_speedup"] = result_dict[backend_speedup_key]
            print(
                f"  MXFP8 ({mxfp8_backend}): Roofline={result['mxfp8_roofline_total_time_ms']:.3f}ms, Actual={mxfp8_actual_ms:.3f}ms"
            )
            if result_dict[backend_speedup_key]:
                print(
                    f"  Actual Speedup ({mxfp8_backend}): {result_dict[backend_speedup_key]:.3f}x"
                )

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

        cutedsl_1x32_time_us = benchmark_mxfp8_quantize_2d_1x32_cutedsl(tensor)
        cutedsl_1x32_bandwidth_gbs = total_gb / (cutedsl_1x32_time_us / 1e6)
        result_dict["mxfp8_quantize_2d_1x32_cutedsl_us"] = cutedsl_1x32_time_us
        result_dict["cutedsl_1x32_bandwidth_gbs"] = cutedsl_1x32_bandwidth_gbs
        result_dict["cutedsl_1x32_efficiency_pct"] = (
            cutedsl_1x32_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mxfp8_quantize_2d_1x32_cutedsl: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cutedsl_1x32_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cutedsl_1x32_efficiency_pct']:.1f}%"
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

        cutedsl_32x1_time_us = benchmark_mxfp8_quantize_2d_32x1_cutedsl(tensor)
        cutedsl_32x1_bandwidth_gbs = total_gb / (cutedsl_32x1_time_us / 1e6)
        result_dict["mxfp8_quantize_2d_32x1_cutedsl_us"] = cutedsl_32x1_time_us
        result_dict["cutedsl_32x1_bandwidth_gbs"] = cutedsl_32x1_bandwidth_gbs
        result_dict["cutedsl_32x1_efficiency_pct"] = (
            cutedsl_32x1_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mxfp8_quantize_2d_32x1_cutedsl: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cutedsl_32x1_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cutedsl_32x1_efficiency_pct']:.1f}%"
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
    print("3D QUANTIZATION KERNELS (Direct Transposed-Weight Quantization)")
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

        # Benchmark the direct grouped-GEMM weight contract: w_t has shape
        # (G, K, N), and the existing 3D 32x1 kernel quantizes it directly.
        weight = torch.randn(G_val, N_val, K_val, dtype=torch.bfloat16, device="cuda")
        tensor = weight.transpose(-2, -1)

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
        input_group_offsets = generate_jagged_offs(num_groups, M, multiple_of=128)

        # Benchmark CUDA kernel
        cuda_out = mx_block_rearrange_2d_M_groups_cuda(
            input_tensor, input_group_offsets
        )
        cuda_time_us = benchmark_cuda_function_in_microseconds(
            mx_block_rearrange_2d_M_groups_cuda,
            input_tensor,
            input_group_offsets,
        )
        cuda_bandwidth_gbs = total_gb / (cuda_time_us / 1e6)
        result_dict["mx_block_rearrange_2d_M_groups_cuda_us"] = cuda_time_us
        result_dict["cuda_bandwidth_gbs"] = cuda_bandwidth_gbs
        result_dict["cuda_efficiency_pct"] = (
            cuda_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mx_block_rearrange_2d_M_groups_cuda: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cuda_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cuda_efficiency_pct']:.1f}%"
        )

        cutedsl_out = _mx_block_rearrange_2d_m_groups_cutedsl(
            input_tensor,
            input_group_offsets,
        )
        torch.testing.assert_close(cutedsl_out, cuda_out, rtol=0, atol=0)
        cutedsl_time_us = benchmark_cuda_function_in_microseconds(
            _mx_block_rearrange_2d_m_groups_cutedsl,
            input_tensor,
            input_group_offsets,
        )
        cutedsl_bandwidth_gbs = total_gb / (cutedsl_time_us / 1e6)
        result_dict["mx_block_rearrange_2d_m_groups_cutedsl_us"] = cutedsl_time_us
        result_dict["cutedsl_bandwidth_gbs"] = cutedsl_bandwidth_gbs
        result_dict["cutedsl_efficiency_pct"] = (
            cutedsl_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mx_block_rearrange_2d_m_groups_cutedsl: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cutedsl_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cutedsl_efficiency_pct']:.1f}%"
        )

        rearrange_results.append(result_dict)

        # Clean up tensors
        del input_tensor, input_group_offsets, cuda_out, cutedsl_out
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
        scale_group_offsets = (
            generate_jagged_offs(num_groups, M, multiple_of=128) // block_size
        )

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

        cutedsl_out = _mx_block_rearrange_2d_k_groups_cutedsl(
            input_tensor,
            scale_group_offsets,
        )
        torch.testing.assert_close(cutedsl_out, triton_out, rtol=0, atol=0)
        cutedsl_time_us = benchmark_cuda_function_in_microseconds(
            _mx_block_rearrange_2d_k_groups_cutedsl,
            input_tensor,
            scale_group_offsets,
        )
        cutedsl_bandwidth_gbs = total_gb / (cutedsl_time_us / 1e6)
        result_dict["mx_block_rearrange_2d_k_groups_cutedsl_us"] = cutedsl_time_us
        result_dict["cutedsl_bandwidth_gbs"] = cutedsl_bandwidth_gbs
        result_dict["cutedsl_efficiency_pct"] = (
            cutedsl_bandwidth_gbs / roofline_bandwidth_gbs
        ) * 100
        print(
            f"  mx_block_rearrange_2d_k_groups_cutedsl: Roofline={roofline_bandwidth_gbs:.1f} GB/s, Actual={cutedsl_bandwidth_gbs:.1f} GB/s, Efficiency={result_dict['cutedsl_efficiency_pct']:.1f}%"
        )

        rearrange_results.append(result_dict)

        # Clean up tensors
        del input_tensor, scale_group_offsets, triton_out, cutedsl_out
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
        offs = generate_jagged_offs(G_val, M, multiple_of=128)

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

        # Convert activations to MXFP8 format using triton_to_mxfp8_dim0
        x_fp8, x_scales = triton_to_mxfp8_dim0(x, inner_block_size=32)
        w_fp8, w_scales_blocked = mxfp8_quantize_cuda_3d(
            w_t,
            block_size=32,
            scale_block_dim1=32,
            scale_block_dim2=1,
            scaling_mode="rceil",
        )

        # Convert only activation scales to blocked format. Weight scales are
        # already produced in blocked layout by mxfp8_quantize_cuda_3d.
        x_scales_blocked, _ = torch_to_blocked_2d_M_groups(
            x_scales, offs, block_size=32
        )

        # Benchmark the MXFP8 grouped GEMM kernel
        mxfp8_gemm_time_us = benchmark_mxfp8_grouped_gemm(
            x_fp8, w_fp8, x_scales_blocked, w_scales_blocked, offs
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
        del x, w, w_t, offs, x_fp8, x_scales, w_fp8
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
        offs = generate_jagged_offs(G_val, M, multiple_of=128)

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
            kernel_preference=None,
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
            kernel_preference=None,
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
    if "actual_speedup" in df_speedup:
        actual_speedup_columns = [("actual_speedup", "Actual Implementation")]
    else:
        actual_speedup_columns = [
            (column, column.removeprefix("actual_speedup_"))
            for column in df_speedup.columns
            if column.startswith("actual_speedup_")
        ]
    for column, label in actual_speedup_columns:
        ax1.plot(
            df_speedup["M"],
            df_speedup[column],
            marker="s",
            linewidth=2,
            linestyle="-",
            label=label,
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
        label="legacy dim0 quant (Triton)",
        color="blue",
    )
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["cutedsl_1x32_efficiency_pct"],
        marker="o",
        linewidth=2,
        linestyle="-",
        label="CuTeDSL 1x32 quant",
        color="green",
    )
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["cuda_dim1_efficiency_pct"],
        marker="d",
        linewidth=2,
        linestyle="-",
        label="legacy dim1 quant (CUDA)",
        color="orange",
    )
    ax2.plot(
        df_quant_2d["M"],
        df_quant_2d["cutedsl_32x1_efficiency_pct"],
        marker="x",
        linewidth=2,
        linestyle="-",
        label="CuTeDSL 32x1 quant",
        color="brown",
    )
    # 2D Rearrange kernels
    df_m_groups = df_rearrange[df_rearrange["kernel_type"] == "M_groups"]
    df_k_groups = df_rearrange[df_rearrange["kernel_type"] == "K_groups"]
    ax2.plot(
        df_m_groups["M"],
        df_m_groups["cuda_efficiency_pct"],
        marker="^",
        linewidth=2,
        linestyle="--",
        label="legacy M-groups layout (CUDA)",
        color="purple",
    )
    ax2.plot(
        df_m_groups["M"],
        df_m_groups["cutedsl_efficiency_pct"],
        marker="v",
        linewidth=2,
        linestyle="--",
        label="CuTeDSL M-groups layout",
        color="gray",
    )
    ax2.plot(
        df_k_groups["M"],
        df_k_groups["triton_efficiency_pct"],
        marker="d",
        linewidth=2,
        linestyle="--",
        label="legacy K-groups layout (Triton)",
        color="red",
    )
    ax2.plot(
        df_k_groups["M"],
        df_k_groups["cutedsl_efficiency_pct"],
        marker="p",
        linewidth=2,
        linestyle="--",
        label="CuTeDSL K-groups layout",
        color="black",
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

    # Plot 4: Backend runtime ratio
    ax4 = axes[0, 2]
    if {
        "mxfp8_actual_time_ms_legacy",
        "mxfp8_actual_time_ms_cutedsl",
    }.issubset(df_speedup.columns):
        backend_runtime_ratio = (
            df_speedup["mxfp8_actual_time_ms_cutedsl"]
            / df_speedup["mxfp8_actual_time_ms_legacy"]
        )
        ax4.plot(
            df_speedup["M"],
            backend_runtime_ratio,
            marker="o",
            linewidth=2,
            linestyle="-",
            label="CuTeDSL / legacy runtime",
            color="green",
        )
        ax4.axhline(
            y=1.0,
            color="red",
            linestyle=":",
            linewidth=1.5,
            label="parity",
        )
        ax4.set_xscale("log", base=2)
        ax4.set_xticks(df_speedup["M"])
        ax4.set_xticklabels([f"{int(m):,}" for m in df_speedup["M"]])
        ratio_min = backend_runtime_ratio.min()
        ratio_max = backend_runtime_ratio.max()
        ratio_margin = max(0.02, (ratio_max - ratio_min) * 0.2)
        ax4.set_ylim(ratio_min - ratio_margin, ratio_max + ratio_margin)
    else:
        ax4.text(
            0.5,
            0.5,
            "Run with --backend=both",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax4.transAxes,
            fontsize=12,
            color="gray",
        )
    ax4.set_xlabel("Local Batch Size x Sequence Length (M)", fontsize=12)
    ax4.set_ylabel("Runtime Ratio", fontsize=12)
    ax4.set_title("Backend Runtime Ratio", fontsize=13)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

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
        label="CuTeDSL 3D quant + blocked scales",
        color="red",
    )
    # 3D Rearrange kernel
    ax5.plot(
        df_rearrange_3d["M"],
        df_rearrange_3d["triton_efficiency_pct"],
        marker="^",
        linewidth=2,
        linestyle="--",
        label="legacy per-group 3D layout (Triton)",
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
    fwd_input_quant_legacy_ms = (
        df_quant_2d.loc[idx_large, "triton_to_mxfp8_dim0_us"] / 1000
    )
    fwd_input_quant_cutedsl_ms = (
        df_quant_2d.loc[idx_large, "mxfp8_quantize_2d_1x32_cutedsl_us"] / 1000
    )

    # Weight quantization: use mxfp8_quantize_cuda_3d directly on w_t, shape
    # (G, K, N), with no separate 3D scale rearrangement step.
    idx_3d_large = df_quant_3d[df_quant_3d["description"] == f"M={M_large}"].index[0]
    fwd_weight_quant_cutedsl_ms = (
        df_quant_3d.loc[idx_3d_large, "mxfp8_quantize_cuda_3d_us"] / 1000
    )
    weight_tensor = torch.randn(
        G_val,
        N_val,
        K_val,
        dtype=torch.bfloat16,
        device="cuda",
    )
    fwd_weight_quant_legacy_ms = benchmark_triton_to_mxfp8_dim0(weight_tensor) / 1000

    # Input scale rearrangement: M-groups for (M, K//32)
    idx_m_groups = df_rearrange[
        (df_rearrange["kernel_type"] == "M_groups") & (df_rearrange["M"] == M_large)
    ].index[0]
    fwd_input_scale_rearrange_legacy_ms = (
        df_rearrange.loc[idx_m_groups, "mx_block_rearrange_2d_M_groups_cuda_us"] / 1000
    )
    fwd_input_scale_rearrange_cutedsl_ms = (
        df_rearrange.loc[idx_m_groups, "mx_block_rearrange_2d_m_groups_cutedsl_us"]
        / 1000
    )

    idx_rearrange_3d_large = df_rearrange_3d[
        df_rearrange_3d["description"] == f"M={M_large}"
    ].index[0]
    fwd_weight_scale_rearrange_legacy_ms = (
        df_rearrange_3d.loc[
            idx_rearrange_3d_large,
            "triton_mx_block_rearrange_per_group_3d_us",
        ]
        / 1000
    )
    fwd_weight_scale_rearrange_cutedsl_ms = 0.0

    # GEMM: use actual MXFP8 2D/3D grouped GEMM time
    idx_gemm = df_grouped_gemm[df_grouped_gemm["M"] == M_large].index[0]
    fwd_gemm_ms = df_grouped_gemm.loc[idx_gemm, "mxfp8_gemm_time_ms"]

    # Backward input pass - need to run additional benchmarks for (M, N) quantization
    # For grad_output quantization (M, N), we can estimate using the 2D kernel with N instead of K
    # Create and benchmark the tensors
    print(f"\nRunning additional benchmarks for kernel breakdown (M={M_large})...")

    # Backward input: grad_output quantization (M, N)
    grad_out_tensor = torch.randn(M_large, N_val, dtype=torch.bfloat16, device="cuda")
    bwd_input_grad_quant_legacy_ms = (
        benchmark_triton_to_mxfp8_dim0(grad_out_tensor) / 1000
    )
    bwd_input_grad_quant_cutedsl_ms = (
        benchmark_mxfp8_quantize_2d_1x32_cutedsl(grad_out_tensor) / 1000
    )

    # Backward input: weight quantization is same as forward (reuse)
    bwd_input_weight_quant_legacy_ms = fwd_weight_quant_legacy_ms
    bwd_input_weight_quant_cutedsl_ms = fwd_weight_quant_cutedsl_ms

    # Backward input: grad scale rearrangement for (M, N//32) - M-groups
    grad_scales = torch.randint(
        0, 256, size=(M_large, N_val // block_size), dtype=torch.uint8, device="cuda"
    )
    grad_offs = generate_jagged_offs(G_val, M_large, multiple_of=128)
    bwd_input_grad_scale_rearrange_legacy_ms = (
        benchmark_cuda_function_in_microseconds(
            mx_block_rearrange_2d_M_groups_cuda, grad_scales, grad_offs
        )
        / 1000
    )
    bwd_input_grad_scale_rearrange_cutedsl_ms = (
        benchmark_cuda_function_in_microseconds(
            _mx_block_rearrange_2d_m_groups_cutedsl, grad_scales, grad_offs
        )
        / 1000
    )

    # Backward input: weight scale rearrangement is same as forward (reuse)
    bwd_input_weight_scale_rearrange_legacy_ms = fwd_weight_scale_rearrange_legacy_ms
    bwd_input_weight_scale_rearrange_cutedsl_ms = fwd_weight_scale_rearrange_cutedsl_ms

    # Backward input: GEMM (same shape as forward, reuse)
    bwd_input_gemm_ms = fwd_gemm_ms

    # Backward weight pass
    bwd_weight_grad_quant_legacy_ms = (
        benchmark_to_mxfp8_dim1_cuda(grad_out_tensor) / 1000
    )
    bwd_weight_grad_quant_cutedsl_ms = (
        benchmark_mxfp8_quantize_2d_32x1_cutedsl(grad_out_tensor) / 1000
    )

    input_tensor_for_wgrad = torch.randn(
        M_large,
        K_val,
        dtype=torch.bfloat16,
        device="cuda",
    )
    bwd_weight_input_quant_legacy_ms = (
        benchmark_to_mxfp8_dim1_cuda(input_tensor_for_wgrad) / 1000
    )
    bwd_weight_input_quant_cutedsl_ms = (
        benchmark_mxfp8_quantize_2d_32x1_cutedsl(input_tensor_for_wgrad) / 1000
    )

    # Grad.T scale rearrangement (N, M//32) - K-groups
    idx_k_groups = df_rearrange[
        (df_rearrange["kernel_type"] == "K_groups") & (df_rearrange["M"] == M_large)
    ].index[0]
    bwd_weight_grad_scale_rearrange_legacy_ms = (
        df_rearrange.loc[idx_k_groups, "triton_mx_block_rearrange_2d_K_groups_us"]
        / 1000
    )
    bwd_weight_grad_scale_rearrange_cutedsl_ms = (
        df_rearrange.loc[idx_k_groups, "mx_block_rearrange_2d_k_groups_cutedsl_us"]
        / 1000
    )

    # Input scale rearrangement - need K-groups for (K, M//32)
    input_scales_k = torch.randint(
        0, 256, size=(K_val, M_large // block_size), dtype=torch.uint8, device="cuda"
    )
    scale_group_offs = (
        generate_jagged_offs(G_val, M_large, multiple_of=128) // block_size
    )
    bwd_weight_input_scale_rearrange_legacy_ms = (
        benchmark_cuda_function_in_microseconds(
            triton_mx_block_rearrange_2d_K_groups, input_scales_k, scale_group_offs
        )
        / 1000
    )
    bwd_weight_input_scale_rearrange_cutedsl_ms = (
        benchmark_cuda_function_in_microseconds(
            _mx_block_rearrange_2d_k_groups_cutedsl, input_scales_k, scale_group_offs
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
        input_tensor_for_wgrad,
        weight_tensor,
        input_scales_k,
        scale_group_offs,
    )
    torch.cuda.empty_cache()

    # Data for stacked bars
    passes = ["Forward", "Backward\nInput", "Backward\nWeight"]

    breakdown = {
        "legacy": {
            "quant_1": [
                fwd_input_quant_legacy_ms,
                bwd_input_grad_quant_legacy_ms,
                bwd_weight_grad_quant_legacy_ms,
            ],
            "quant_2": [
                fwd_weight_quant_legacy_ms,
                bwd_input_weight_quant_legacy_ms,
                bwd_weight_input_quant_legacy_ms,
            ],
            "rearrange_1": [
                fwd_input_scale_rearrange_legacy_ms,
                bwd_input_grad_scale_rearrange_legacy_ms,
                bwd_weight_grad_scale_rearrange_legacy_ms,
            ],
            "rearrange_2": [
                fwd_weight_scale_rearrange_legacy_ms,
                bwd_input_weight_scale_rearrange_legacy_ms,
                bwd_weight_input_scale_rearrange_legacy_ms,
            ],
        },
        "cutedsl": {
            "quant_1": [
                fwd_input_quant_cutedsl_ms,
                bwd_input_grad_quant_cutedsl_ms,
                bwd_weight_grad_quant_cutedsl_ms,
            ],
            "quant_2": [
                fwd_weight_quant_cutedsl_ms,
                bwd_input_weight_quant_cutedsl_ms,
                bwd_weight_input_quant_cutedsl_ms,
            ],
            "rearrange_1": [
                fwd_input_scale_rearrange_cutedsl_ms,
                bwd_input_grad_scale_rearrange_cutedsl_ms,
                bwd_weight_grad_scale_rearrange_cutedsl_ms,
            ],
            "rearrange_2": [
                fwd_weight_scale_rearrange_cutedsl_ms,
                bwd_input_weight_scale_rearrange_cutedsl_ms,
                bwd_weight_input_scale_rearrange_cutedsl_ms,
            ],
        },
    }
    gemm = [fwd_gemm_ms, bwd_input_gemm_ms, bwd_weight_gemm_ms]

    # Create paired stacked bars
    group_centers = list(range(len(passes)))
    bar_width = 0.32
    backend_offsets = {"legacy": -bar_width / 2, "cutedsl": bar_width / 2}
    component_specs = [
        ("quant_1", "Input/Grad Quant", "#1f77b4"),
        ("quant_2", "Weight Quant", "#ff7f0e"),
        ("rearrange_1", "Input/Grad Rearrange", "#2ca02c"),
        ("rearrange_2", "Weight Rearrange", "#d62728"),
        ("gemm", "GEMM", "#9467bd"),
    ]
    x_by_backend = {
        backend_name: [center + offset for center in group_centers]
        for backend_name, offset in backend_offsets.items()
    }
    totals_by_backend = {}
    for backend_name, x_values in x_by_backend.items():
        bottoms = [0.0, 0.0, 0.0]
        totals_by_backend[backend_name] = [0.0, 0.0, 0.0]
        for component_name, label, color in component_specs:
            values = (
                gemm
                if component_name == "gemm"
                else breakdown[backend_name][component_name]
            )
            ax6.bar(
                x_values,
                values,
                bar_width,
                bottom=bottoms,
                label=label if backend_name == "legacy" else None,
                color=color,
                alpha=0.70 if backend_name == "legacy" else 1.0,
            )
            bottoms = [bottoms[i] + values[i] for i in range(3)]
            totals_by_backend[backend_name] = bottoms

    # Formatting
    ax6.set_ylabel("Time (ms)", fontsize=12)
    ax6.set_title(f"Kernel Breakdown (M={M_large:,})", fontsize=13)
    ax6.set_xticks(group_centers)
    ax6.set_xticklabels(passes, fontsize=10)
    ax6.grid(True, alpha=0.3, axis="y")
    for backend_name, x_values in x_by_backend.items():
        for x_value in x_values:
            ax6.text(
                x_value,
                -0.04,
                backend_name,
                ha="center",
                va="top",
                fontsize=8,
                rotation=45,
                transform=ax6.get_xaxis_transform(),
            )

    # Add total time labels on top of each bar
    for backend_name, x_values in x_by_backend.items():
        for pos, total in zip(x_values, totals_by_backend[backend_name]):
            ax6.text(pos, total, f"{total:.1f}", ha="center", va="bottom", fontsize=8)

    # Add BF16 GEMM baseline reference lines
    # Forward and Backward Input use 2D/3D GEMM
    bf16_fwd_gemm_ms = df_grouped_gemm.loc[idx_gemm, "bf16_gemm_time_ms"]
    # Backward Weight uses 2D/2D GEMM
    bf16_bwd_weight_gemm_ms = df_grouped_gemm_2d_2d.loc[
        idx_gemm_2d2d, "bf16_2d_2d_gemm_time_ms"
    ]

    # Draw horizontal lines for BF16 baseline at each bar position
    bar_width_visual = 0.42
    for i, baseline_ms in enumerate(
        [bf16_fwd_gemm_ms, bf16_fwd_gemm_ms, bf16_bwd_weight_gemm_ms]
    ):
        label = "BF16 GEMM baseline" if i == 0 else None
        ax6.plot(
            [group_centers[i] - bar_width_visual, group_centers[i] + bar_width_visual],
            [baseline_ms, baseline_ms],
            color="red",
            linestyle="--",
            linewidth=2,
            label=label,
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
    if "actual_speedup" in df_speedup:
        actual_speedup_summary = (
            f"  Average actual speedup: {df_speedup['actual_speedup'].mean():.3f}x\n"
            f"  Median actual speedup: {df_speedup['actual_speedup'].median():.3f}x"
        )
    else:
        summary_lines = []
        for column in df_speedup.columns:
            if column.startswith("actual_speedup_"):
                backend_name = column.removeprefix("actual_speedup_")
                summary_lines.append(
                    f"  Average actual speedup ({backend_name}): {df_speedup[column].mean():.3f}x"
                )
                summary_lines.append(
                    f"  Median actual speedup ({backend_name}): {df_speedup[column].median():.3f}x"
                )
        actual_speedup_summary = "\n".join(summary_lines)
    print(
        f"""
Net Speedup Analysis:
  Average roofline speedup: {df_speedup["roofline_speedup"].mean():.3f}x
{actual_speedup_summary}

2D Quantization Kernels:
  triton_to_mxfp8_dim0 avg efficiency: {df_quant_2d["triton_dim0_efficiency_pct"].mean():.1f}%
  mxfp8_quantize_2d_1x32_cutedsl avg efficiency: {df_quant_2d["cutedsl_1x32_efficiency_pct"].mean():.1f}%
  to_mxfp8_dim1_cuda avg efficiency: {df_quant_2d["cuda_dim1_efficiency_pct"].mean():.1f}%
  mxfp8_quantize_2d_32x1_cutedsl avg efficiency: {df_quant_2d["cutedsl_32x1_efficiency_pct"].mean():.1f}%

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
  Achievable BW roofline: {model.memory_bandwidth_gbs:.1f} GB/s
  Achievable BF16 roofline: {model.bf16_tflops:.1f} TFLOPS
  Achievable MXFP8 roofline: {model.mxfp8_tflops:.1f} TFLOPS
"""
    )
    print("=" * 80)


if __name__ == "__main__":
    fire.Fire(run)

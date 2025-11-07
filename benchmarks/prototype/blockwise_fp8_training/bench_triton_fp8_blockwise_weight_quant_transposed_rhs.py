# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import List, Tuple

import torch
from tabulate import tabulate
from tqdm import tqdm

# Assuming these imports based on the kernel location
from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)

device = torch.device("cuda")

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    input_shape: Tuple[int, int]  # (M, N)
    block_size: int


@dataclass(frozen=True)
class ExperimentResult:
    # time
    naive_us: float
    triton_us: float
    # mem bw
    naive_gbps: float
    triton_gbps: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    """
    Test configurations for typical weight matrix shapes.
    Format: (hidden_dim, hidden_dim) for square matrices or (hidden_dim_in, hidden_dim_out)

    Note: Both M and N must be divisible by block_size (128)
    """
    # Common weight matrix shapes in transformers
    # Format: (in_features, out_features) for weight matrices
    input_shapes = [
        (512, 4096),
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
        (8192, 4096),
    ]

    configs = []
    block_sizes = [128]  # Standard block size for FP8

    for shape in input_shapes:
        for block_size in block_sizes:
            # Verify both dimensions are divisible by block_size
            if shape[0] % block_size == 0 and shape[1] % block_size == 0:
                configs.append(
                    ExperimentConfig(
                        input_shape=shape,
                        block_size=block_size,
                    )
                )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run benchmark experiment comparing naive and Triton implementations.
    """
    M, N = config.input_shape
    block_size = config.block_size

    def naive_fp8_blockwise_weight_quant_transposed(
        x: torch.Tensor, block_size: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Naive PyTorch reference implementation for blockwise FP8 weight quantization with transpose.

        Quantizes in (block_size x block_size) blocks. Each block gets one scale factor.
        Outputs transposed tensor (N, M) in column-major format for RHS operator.

        Args:
            x: Input tensor of shape (M, N), row-major
            block_size: Block size for quantization

        Returns:
            y: Transposed quantized tensor in FP8, shape (N, M), column-major format
            s: Reciprocal scales in column-major format (N//block_size, M//block_size)
        """
        assert x.is_contiguous(), "Input must be contiguous"
        assert x.dim() == 2, "Input must be 2D"
        assert x.size(0) % block_size == 0 and x.size(1) % block_size == 0, (
            "Both dimensions must be divisible by block_size"
        )

        M, N = x.size()
        M_blocks = M // block_size
        N_blocks = N // block_size

        # FP8 E4M3 constants
        max_fp8_e4m3 = 448.0
        min_fp8_e4m3 = -448.0
        eps = 1e-12

        # Reshape to (M_blocks, block_size, N_blocks, block_size)
        x_reshaped = x.view(M_blocks, block_size, N_blocks, block_size)
        # Permute to (M_blocks, N_blocks, block_size, block_size) for easier block processing
        x_blocks = x_reshaped.permute(0, 2, 1, 3)

        # Compute max absolute value per block (M_blocks, N_blocks)
        amax = torch.clamp(
            x_blocks.abs().amax(dim=(2, 3)).to(torch.float64), min=eps, max=float("inf")
        )

        # Compute scales (M_blocks, N_blocks)
        scale = (max_fp8_e4m3 / amax).to(torch.float32)

        # Broadcast scale for quantization (M_blocks, N_blocks, 1, 1)
        scale_broadcast = scale[:, :, None, None]

        # Quantize
        y_blocks = x_blocks * scale_broadcast
        y_blocks = torch.clamp(y_blocks, min=min_fp8_e4m3, max=max_fp8_e4m3)

        # Permute back: (M_blocks, N_blocks, block_size, block_size) -> (M_blocks, block_size, N_blocks, block_size)
        y_reshaped = y_blocks.permute(0, 2, 1, 3)
        # Reshape to (M, N) then transpose to (N, M)
        y_rowmajor = y_reshaped.reshape(
            M, N).t()

        # Convert to FP8 and create column-major output (matching Triton kernel)
        y = y_rowmajor.t().contiguous().t()
        y = y_rowmajor.to(torch.float8_e4m3fn)

        # Compute reciprocal scales
        reciprocal_scale = 1.0 / scale
        # Transpose scale matrix to match output dimensions: (M_blocks, N_blocks) -> (N_blocks, M_blocks)
        reciprocal_scale = reciprocal_scale.t()

        # Convert to col-major
        s = reciprocal_scale.t().contiguous().t()

        return y, s

    def verify_outputs(
        y_naive: torch.Tensor,
        s_naive: torch.Tensor,
        y_triton: torch.Tensor,
        s_triton: torch.Tensor,
        rtol: float = 1e-2,
        atol: float = 1e-2,
    ):
        """Verify that Triton and naive implementations produce similar results."""

        # Convert FP8 back to float for comparison

        y_naive_float = y_naive.to(torch.float32)
        y_triton_float = y_triton.to(torch.float32)

        # Check quantized values are close
        torch.testing.assert_close(
            y_naive_float,
            y_triton_float,
            rtol=rtol,
            atol=atol,
            msg="Quantized values differ between naive and Triton implementations",
        )

        torch.testing.assert_close(
            s_naive,
            s_triton,
            rtol=rtol,
            atol=atol,
            msg="Scales differ between naive and Triton implementations",
        )

    # Create input tensor
    input_tensor = torch.randn(
        M,
        N,
        dtype=torch.bfloat16,
        device=device,
    )

    # Benchmark naive implementation (torch.compile handles warmup)
    naive_impl_c = torch.compile(naive_fp8_blockwise_weight_quant_transposed)
    y_naive, s_naive = naive_impl_c(input_tensor, block_size)
    naive_time_us = benchmark_cuda_function_in_microseconds(
        naive_impl_c,
        input_tensor,
        block_size,
    )

    # Benchmark Triton implementation (torch.compile handles warmup)
    y_triton, s_triton = triton_fp8_blockwise_weight_quant_transposed_rhs(
        input_tensor, block_size)
    triton_time_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_blockwise_weight_quant_transposed_rhs,
        input_tensor,
        block_size,
    )

    # Verify correctness
    verify_outputs(y_naive, s_naive, y_triton, s_triton)

    # Memory bandwidth calculations
    bytes_per_input_el = torch.finfo(input_tensor.dtype).bits / 8
    bytes_per_output_el = torch.finfo(y_triton.dtype).bits / 8
    bytes_per_scale_el = torch.finfo(s_triton.dtype).bits / 8

    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = (
        y_triton.numel() * bytes_per_output_el + s_triton.numel() * bytes_per_scale_el
    )

    naive_gbps = ((read_bytes + write_bytes) / 1e9) / (naive_time_us / 1e6)
    triton_gbps = ((read_bytes + write_bytes) / 1e9) / (triton_time_us / 1e6)

    return ExperimentResult(
        naive_us=naive_time_us,
        triton_us=triton_time_us,
        naive_gbps=naive_gbps,
        triton_gbps=triton_gbps,
    )


def print_results(experiments: List[Experiment]):
    """Print benchmark results in a formatted table."""
    headers = [
        "input_shape (M, N)",
        "block_size",
        "naive_us",
        "triton_us",
        "speedup",
        "naive_gbps",
        "triton_gbps",
    ]
    rows = []
    for experiment in experiments:
        speedup = experiment.result.naive_us / experiment.result.triton_us
        rows.append(
            [
                f"{experiment.config.input_shape[0]}x{experiment.config.input_shape[1]}",
                experiment.config.block_size,
                f"{experiment.result.naive_us:.2f}",
                f"{experiment.result.triton_us:.2f}",
                f"{speedup:.2f}x",
                f"{experiment.result.naive_gbps:.1f}",
                f"{experiment.result.triton_gbps:.1f}",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def main():
    """Main benchmark execution."""
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []

    print(f"Running {len(configs)} benchmark configurations...\n")

    for config in tqdm(configs, desc="Benchmarking"):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS - Transposed RHS Weight Quantization")
    print("=" * 80 + "\n")
    print_results(results)


if __name__ == "__main__":
    main()

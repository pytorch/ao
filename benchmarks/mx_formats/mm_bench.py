import itertools
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from jsonargparse import CLI
from tabulate import tabulate
from torch._inductor.utils import do_bench_using_profiling
from tqdm import tqdm

from torchao.ops import mx_fp4_bf16
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.prototype.mx_formats.utils import to_blocked


class Format(Enum):
    MX_FP8 = "MX-FP8"
    MX_FP4 = "MX-FP4"


def get_mx_matmul(A: torch.Tensor, B: torch.Tensor, format: Format):
    if format == Format.MX_FP8:
        dtype = torch.float8_e4m3fn
        fn = partial(torch._scaled_mm, out_dtype=torch.bfloat16)
    elif format == Format.MX_FP4:
        dtype = torch.float4_e2m1fn_x2
        fn = mx_fp4_bf16
    else:
        raise ValueError(f"Invalid format: {format}")

    a_scale, A_lp = to_mx(A, dtype, 32)
    b_scale, B_lp_t = to_mx(B.T, dtype, 32)
    assert B_lp_t.is_contiguous()
    B_lp = B_lp_t.T

    a_scale = to_blocked(a_scale.view(A.shape[0], A.shape[1] // 32))
    b_scale = to_blocked(b_scale.view(B.shape[1], B.shape[0] // 32))

    return lambda: fn(A_lp, B_lp, a_scale, b_scale)


@dataclass(frozen=True)
class ExperimentConfig:
    M: int
    K: int
    N: int
    format: Format


@dataclass(frozen=True)
class ExperimentResult:
    time: float
    tflops: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def calculate_tflops(M: int, N: int, K: int, time_us: float) -> float:
    """Calculate TFLOPS (Tera Floating Point Operations Per Second)"""
    # Number of floating point operations for matrix multiplication
    flops = 2 * M * N * K
    tflops = (flops / time_us) / 1e6  # Convert to TFLOPS
    return tflops


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    A = torch.zeros(config.M, config.K, device="cuda", dtype=torch.bfloat16)
    B = torch.zeros(config.N, config.K, device="cuda", dtype=torch.bfloat16).T

    matmul = get_mx_matmul(A, B, config.format)

    # Warmup phase
    warmup_iterations = 5
    for _ in range(warmup_iterations):
        _ = matmul()
    torch.cuda.synchronize()

    # Actual benchmarking
    time_us = do_bench_using_profiling(matmul) * 1e3
    tflops = calculate_tflops(config.M, config.N, config.K, time_us)

    return ExperimentResult(time=time_us, tflops=tflops)


def print_results(experiments: list[Experiment], save_path: Path | None = None):
    headers = ["M", "K", "N", "Format", "Time (ms)", "TFLOPS"]
    rows = []
    for experiment in experiments:
        config = experiment.config
        result = experiment.result

        rows.append(
            [
                config.M,
                config.K,
                config.N,
                config.format.value,
                f"{result.time:.4f}",
                f"{result.tflops:.2f}",
            ]
        )

    print(tabulate(rows, headers=headers))

    if save_path is not None:
        pd.DataFrame(rows, columns=headers).to_csv(save_path, index=False)
        print(f"💾 Results saved to: {save_path}")


def plot_tflops_comparison(df, save_path: Path):
    plt.figure(figsize=(12, 6))
    grouped = df.groupby(["K", "Format"])
    k_values = sorted(df["K"].unique())
    formats = df["Format"].unique()
    m_value = df["M"].iloc[0]
    n_value = df["N"].iloc[0]

    # Plot MX kernel performance
    for format in formats:
        try:
            tflops_values = [
                grouped.get_group((k, format))["TFLOPS"].values[0] for k in k_values
            ]
            plt.plot(k_values, tflops_values, label=format)
        except KeyError:
            # Skip if this combination doesn't exist in the data
            continue

    plt.xlabel("K (Matrix Dimension)")
    plt.ylabel("TFLOPS")

    # Set y-axis to start at 0
    plt.ylim(bottom=0)

    title = f"MX Matrix Multiplication Performance \nM={m_value}, N={n_value}"
    plt.title(title)

    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xticks(k_values, rotation=45, ha="right")
    plt.tight_layout()

    # Generate the file name and save in the same directory as the CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"mx_{m_value}_{n_value}_{timestamp}.png"
    graph_path = save_path.parent / file_name
    plt.savefig(graph_path, dpi=300)
    print(f"TFLOPS comparison plot saved as {graph_path}")


def get_configs_varying_k(M: int = 8192, N: int = 8192) -> list[ExperimentConfig]:
    shapes = [(M, K, N) for K in range(1024, 16385, 1024)]
    formats = [Format.MX_FP8, Format.MX_FP4]

    configs = [
        ExperimentConfig(M=M, K=K, N=N, format=format)
        for (M, K, N), format in itertools.product(shapes, formats)
    ]
    return configs


def main(
    save_path: str | None = None, M: int = 8192, N: int = 8192, graph: bool = False
):
    """Benchmark MX MatMul with different configurations and optionally graph results.

    Args:
        save_path (Optional[str], optional): Path to save the results. Defaults to None.
        M (int, optional): Number of rows in the first matrix. Defaults to 8192.
        N (int, optional): Number of columns in the second matrix. Defaults to 8192.
        graph (bool, optional): Whether to create a graph of the results. Defaults to False.
    """
    torch.random.manual_seed(123)
    configs = get_configs_varying_k(M, N)
    results = []
    if save_path is not None:
        save_path = Path(save_path)
        save_path = save_path.with_suffix(".csv")
        save_path.parent.mkdir(parents=True, exist_ok=True)
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))
    print_results(results, save_path)

    if graph and save_path is not None:
        df = pd.read_csv(save_path)
        plot_tflops_comparison(df, save_path)


if __name__ == "__main__":
    CLI(main)

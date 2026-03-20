# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from tabulate import tabulate

from benchmarks.utils import benchmark_cuda_function_in_microseconds
from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_act_quant_rhs,
    triton_fp8_blockwise_act_quant_transposed_lhs,
    triton_fp8_blockwise_weight_quant_rhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)
from torchao.testing.training.roofline_utils import gpu_name_to_specs


@dataclass(frozen=True)
class KernelMeasurement:
    kernel: str
    shape: Tuple[int, int]
    kernel_us: float
    effective_logical_io_gbps: float
    logical_io_vs_peak_pct: float
    logical_io_vs_achievable_pct: Optional[float]


@dataclass(frozen=True)
class SkippedKernelCase:
    kernel: str
    shape: Tuple[int, int]
    reason: str


@dataclass(frozen=True)
class GpuBandwidthSpec:
    gpu_name: str
    peak_gbps: float
    peak_source: str
    achievable_gbps: Optional[float]
    achievable_source: Optional[str]


@dataclass(frozen=True)
class KernelSpec:
    name: str
    runner: Callable[[torch.Tensor, int], Tuple[torch.Tensor, torch.Tensor]]
    validate: Callable[[Tuple[int, int], int], Optional[str]]


def _validate_k_divisible(shape: Tuple[int, int], block_size: int) -> Optional[str]:
    _, k = shape
    if k % block_size != 0:
        return f"K={k} must be divisible by block_size={block_size}"
    return None


def _validate_m_divisible(shape: Tuple[int, int], block_size: int) -> Optional[str]:
    m, _ = shape
    if m % block_size != 0:
        return f"M={m} must be divisible by block_size={block_size}"
    return None


def _validate_mk_divisible(shape: Tuple[int, int], block_size: int) -> Optional[str]:
    for validator in (_validate_m_divisible, _validate_k_divisible):
        reason = validator(shape, block_size)
        if reason is not None:
            return reason
    return None


KERNEL_SPECS = [
    KernelSpec(
        name="act_quant_lhs",
        runner=triton_fp8_blockwise_act_quant_lhs,
        validate=_validate_k_divisible,
    ),
    KernelSpec(
        name="act_quant_rhs",
        runner=triton_fp8_blockwise_act_quant_rhs,
        validate=_validate_k_divisible,
    ),
    KernelSpec(
        name="act_quant_transposed_lhs",
        runner=triton_fp8_blockwise_act_quant_transposed_lhs,
        validate=_validate_m_divisible,
    ),
    KernelSpec(
        name="weight_quant_rhs",
        runner=triton_fp8_blockwise_weight_quant_rhs,
        validate=_validate_mk_divisible,
    ),
    KernelSpec(
        name="weight_quant_transposed_rhs",
        runner=triton_fp8_blockwise_weight_quant_transposed_rhs,
        validate=_validate_mk_divisible,
    ),
]


def _lookup_roofline_specs(gpu_name: str) -> Optional[dict]:
    specs = gpu_name_to_specs.get(gpu_name)
    if specs is not None:
        return specs
    for known_name, candidate in gpu_name_to_specs.items():
        if known_name in gpu_name or gpu_name in known_name:
            return candidate
    return None


def _resolve_gpu_specs(use_roofline_utils: bool = False) -> GpuBandwidthSpec:
    gpu_name = torch.cuda.get_device_name(0)
    specs = _lookup_roofline_specs(gpu_name)

    if use_roofline_utils:
        if specs is None:
            raise ValueError(f"Unsupported GPU for roofline lookup: {gpu_name}")
        peak_gbps = specs["peak_mem_bw_bytes_sec"] / 1e9
        peak_source = "roofline_utils"
    else:
        peak_mem_bw_bytes_sec = _get_peak_mem_bw_from_device_properties()
        if peak_mem_bw_bytes_sec is not None:
            peak_gbps = peak_mem_bw_bytes_sec / 1e9
            peak_source = "cuda_device_properties"
        elif specs is not None:
            peak_gbps = specs["peak_mem_bw_bytes_sec"] / 1e9
            peak_source = "roofline_utils_fallback"
        else:
            raise ValueError(f"Unsupported GPU for roofline lookup: {gpu_name}")

    if specs is not None and "pct_achievable_mem_bw" in specs:
        achievable_gbps = peak_gbps * specs["pct_achievable_mem_bw"]
        achievable_source = "roofline_utils_pct_achievable_mem_bw"
    else:
        achievable_gbps = None
        achievable_source = None

    return GpuBandwidthSpec(
        gpu_name=gpu_name,
        peak_gbps=peak_gbps,
        peak_source=peak_source,
        achievable_gbps=achievable_gbps,
        achievable_source=achievable_source,
    )


def _get_peak_mem_bw_from_device_properties() -> Optional[float]:
    props = torch.cuda.get_device_properties(0)
    memory_clock_khz = getattr(props, "memory_clock_rate", 0)
    memory_bus_width_bits = getattr(props, "memory_bus_width", 0)
    if memory_clock_khz <= 0 or memory_bus_width_bits <= 0:
        return None

    return (memory_bus_width_bits / 8.0) * (memory_clock_khz * 1e3) * 2.0


def _benchmark_kernel(
    kernel: KernelSpec, input_tensor: torch.Tensor, block_size: int
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    y, s = kernel.runner(input_tensor, block_size)
    kernel_us = benchmark_cuda_function_in_microseconds(
        kernel.runner,
        input_tensor,
        block_size,
    )
    return kernel_us, y, s


def _calculate_logical_io_gbps(
    input_tensor: torch.Tensor,
    y: torch.Tensor,
    s: torch.Tensor,
    kernel_us: float,
) -> float:
    bytes_per_input_el = torch.finfo(input_tensor.dtype).bits / 8
    bytes_per_output_el = torch.finfo(y.dtype).bits / 8
    bytes_per_scale_el = torch.finfo(s.dtype).bits / 8
    read_bytes = input_tensor.numel() * bytes_per_input_el
    write_bytes = y.numel() * bytes_per_output_el + s.numel() * bytes_per_scale_el
    return ((read_bytes + write_bytes) / 1e9) / (kernel_us / 1e6)


def _run_suite(
    m_values: Iterable[int],
    k: int,
    block_size: int,
    bandwidth_spec: GpuBandwidthSpec,
) -> Tuple[List[KernelMeasurement], List[SkippedKernelCase]]:
    measurements = []
    skipped = []

    for m in m_values:
        shape = (m, k)
        for kernel in KERNEL_SPECS:
            reason = kernel.validate(shape, block_size)
            if reason is not None:
                skipped.append(
                    SkippedKernelCase(
                        kernel=kernel.name,
                        shape=shape,
                        reason=reason,
                    )
                )
                continue

            input_tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
            kernel_us, y, s = _benchmark_kernel(kernel, input_tensor, block_size)
            effective_logical_io_gbps = _calculate_logical_io_gbps(
                input_tensor=input_tensor,
                y=y,
                s=s,
                kernel_us=kernel_us,
            )
            logical_io_vs_achievable_pct = None
            if bandwidth_spec.achievable_gbps is not None:
                logical_io_vs_achievable_pct = (
                    effective_logical_io_gbps / bandwidth_spec.achievable_gbps
                ) * 100.0
            measurements.append(
                KernelMeasurement(
                    kernel=kernel.name,
                    shape=shape,
                    kernel_us=kernel_us,
                    effective_logical_io_gbps=effective_logical_io_gbps,
                    logical_io_vs_peak_pct=(
                        effective_logical_io_gbps / bandwidth_spec.peak_gbps
                    )
                    * 100.0,
                    logical_io_vs_achievable_pct=logical_io_vs_achievable_pct,
                )
            )
    return measurements, skipped


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}"


def _print_results(
    measurements: List[KernelMeasurement],
    skipped: List[SkippedKernelCase],
    bandwidth_spec: GpuBandwidthSpec,
) -> None:
    if not measurements:
        raise RuntimeError("No valid kernel/shape combinations were benchmarked.")

    print(f"GPU: {bandwidth_spec.gpu_name}")
    print(f"Peak bandwidth reference: {bandwidth_spec.peak_gbps:.1f} GB/s")
    print(f"Peak bandwidth source: {bandwidth_spec.peak_source}")
    if bandwidth_spec.achievable_gbps is not None:
        print(
            f"Achievable bandwidth reference: {bandwidth_spec.achievable_gbps:.1f} GB/s"
        )
        print(f"Achievable bandwidth source: {bandwidth_spec.achievable_source}")
    else:
        print("Achievable bandwidth reference: n/a")
    print("Timing reflects public quantization wrapper calls.")
    print(
        "effective_logical_io_gbps uses modeled tensor IO bytes, not hardware DRAM counters."
    )
    print()

    rows = []
    for measurement in sorted(
        measurements, key=lambda item: (item.shape[0], item.logical_io_vs_peak_pct)
    ):
        rows.append(
            [
                measurement.kernel,
                f"{measurement.shape[0]}x{measurement.shape[1]}",
                f"{measurement.kernel_us:.2f}",
                f"{measurement.effective_logical_io_gbps:.1f}",
                f"{measurement.logical_io_vs_peak_pct:.1f}",
                _format_optional_float(measurement.logical_io_vs_achievable_pct),
            ]
        )
    print(
        tabulate(
            rows,
            headers=[
                "kernel",
                "shape",
                "kernel_us",
                "effective_logical_io_gbps",
                "logical_io_vs_peak_%",
                "logical_io_vs_achievable_%",
            ],
            tablefmt="github",
        )
    )
    print()

    overall_rows = []
    for kernel in KERNEL_SPECS:
        kernel_measurements = [
            item for item in measurements if item.kernel == kernel.name
        ]
        if not kernel_measurements:
            continue
        avg_peak_util = sum(
            item.logical_io_vs_peak_pct for item in kernel_measurements
        ) / len(kernel_measurements)
        min_peak_util = min(item.logical_io_vs_peak_pct for item in kernel_measurements)
        avg_logical_io_gbps = sum(
            item.effective_logical_io_gbps for item in kernel_measurements
        ) / len(kernel_measurements)
        overall_rows.append(
            [
                kernel.name,
                f"{avg_logical_io_gbps:.1f}",
                f"{avg_peak_util:.1f}",
                f"{min_peak_util:.1f}",
            ]
        )
    overall_rows.sort(key=lambda row: float(row[2]))
    print(
        tabulate(
            overall_rows,
            headers=[
                "kernel",
                "avg_effective_logical_io_gbps",
                "avg_logical_io_vs_peak_%",
                "worst_case_logical_io_vs_peak_%",
            ],
            tablefmt="github",
        )
    )

    if skipped:
        print()
        skipped_rows = [
            [item.kernel, f"{item.shape[0]}x{item.shape[1]}", item.reason]
            for item in skipped
        ]
        print(
            tabulate(
                skipped_rows,
                headers=["skipped_kernel", "shape", "reason"],
                tablefmt="github",
            )
        )


def _write_csv(
    path: Path,
    measurements: List[KernelMeasurement],
    bandwidth_spec: GpuBandwidthSpec,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "kernel",
                "m",
                "k",
                "kernel_us",
                "effective_logical_io_gbps",
                "logical_io_vs_peak_pct",
                "logical_io_vs_achievable_pct",
                "peak_bandwidth_gbps",
                "peak_bandwidth_source",
                "achievable_bandwidth_gbps",
                "achievable_bandwidth_source",
            ],
        )
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(
                {
                    "kernel": measurement.kernel,
                    "m": measurement.shape[0],
                    "k": measurement.shape[1],
                    "kernel_us": measurement.kernel_us,
                    "effective_logical_io_gbps": (
                        measurement.effective_logical_io_gbps
                    ),
                    "logical_io_vs_peak_pct": measurement.logical_io_vs_peak_pct,
                    "logical_io_vs_achievable_pct": (
                        measurement.logical_io_vs_achievable_pct
                    ),
                    "peak_bandwidth_gbps": bandwidth_spec.peak_gbps,
                    "peak_bandwidth_source": bandwidth_spec.peak_source,
                    "achievable_bandwidth_gbps": bandwidth_spec.achievable_gbps,
                    "achievable_bandwidth_source": (bandwidth_spec.achievable_source),
                }
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark blockwise FP8 quantization wrappers and report logical IO "
            "bandwidth against device bandwidth references."
        )
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=[32768, 131072],
        help="M dimensions to benchmark.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4096,
        help="Feature dimension used for every benchmarked shape.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block size passed into each kernel benchmark.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--use-roofline-utils",
        action="store_true",
        help=(
            "Use the static peak bandwidth values from roofline_utils instead of "
            "CUDA device properties."
        ),
    )
    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    torch.random.manual_seed(67)
    args = parse_args()
    bandwidth_spec = _resolve_gpu_specs(use_roofline_utils=args.use_roofline_utils)

    measurements, skipped = _run_suite(
        m_values=args.m_values,
        k=args.k,
        block_size=args.block_size,
        bandwidth_spec=bandwidth_spec,
    )
    _print_results(
        measurements=measurements,
        skipped=skipped,
        bandwidth_spec=bandwidth_spec,
    )

    if args.csv is not None:
        _write_csv(args.csv, measurements, bandwidth_spec)
        print()
        print(f"Wrote CSV results to {args.csv}")


if __name__ == "__main__":
    main()
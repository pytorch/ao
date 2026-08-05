# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tabulate import tabulate

from torchao.prototype.blockwise_fp8_training.cutedsl_grouped_gemm import (
    _cutedsl_runtime_available,
    _missing_cutedsl_runtime_packages,
    cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d,
)
from torchao.prototype.blockwise_fp8_training.grouped_kernels import (
    emulated_blockwise_scaled_grouped_mm,
)
from torchao.prototype.blockwise_fp8_training.grouped_weight_quant import (
    triton_fp8_blockwise_weight_quant_grouped_forward_rhs,
)
from torchao.prototype.blockwise_fp8_training.kernels import (
    BLOCKWISE_1X128_SCALING_TYPE,
    BLOCKWISE_128X128_SCALING_TYPE,
    _scaling_type_value,
    triton_fp8_blockwise_act_quant_lhs,
)
from torchao.testing.training.roofline_utils import (
    KERNEL_LAUNCH_OVERHEAD_SEC,
    get_roofline_gpu_name,
    gpu_name_to_specs,
)
from torchao.utils import ceil_div, is_sm_at_least_90

BLOCK_SIZE = 128


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    e: int
    m_per_group: int
    n: int
    k: int

    @property
    def m_total(self) -> int:
        return self.e * self.m_per_group


@dataclass(frozen=True)
class RooflineEstimate:
    fp8_compute_us: float
    fp8_memory_us: float
    fp8_launch_us: float
    fp8_roofline_us: float
    fp8_roofline_bound: str
    fp8_roofline_tflops: float
    bf16_compute_us: float
    logical_io_bytes: int
    arithmetic_intensity_flops_per_byte: float


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported out dtype: {name}")


def _dtype_size(dtype: torch.dtype) -> int:
    if dtype == torch.bfloat16:
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype: {dtype}")


def _validate_config(config: ExperimentConfig):
    if config.e <= 0 or config.m_per_group <= 0 or config.n <= 0 or config.k <= 0:
        raise ValueError(f"shape values must be positive: {config}")
    if config.n % BLOCK_SIZE != 0 or config.k % BLOCK_SIZE != 0:
        raise ValueError(
            f"{config.name}: N={config.n} and K={config.k} must be divisible "
            f"by block_size={BLOCK_SIZE}"
        )


def _parse_shape(shape: str) -> ExperimentConfig:
    parts = [part.strip() for part in shape.split(",")]
    if len(parts) == 4:
        e, m_per_group, n, k = (int(part) for part in parts)
        name = f"custom_e{e}_m{m_per_group}_n{n}_k{k}"
    elif len(parts) == 5:
        name = parts[0]
        e, m_per_group, n, k = (int(part) for part in parts[1:])
    else:
        raise ValueError(
            "--shape must be 'E,M_PER_GROUP,N,K' or 'NAME,E,M_PER_GROUP,N,K'"
        )
    return ExperimentConfig(name, e, m_per_group, n, k)


def _preset_configs(shape_preset: str) -> list[ExperimentConfig]:
    if shape_preset == "smoke":
        return [
            ExperimentConfig("smoke_e2_m256_n256_k256", 2, 256, 256, 256),
            ExperimentConfig("medium_e2_m512_n1024_k1024", 2, 512, 1024, 1024),
        ]
    if shape_preset == "sweep":
        return [
            ExperimentConfig(f"e{e}_m{m}_n{n}_k{k}", e, m, n, k)
            for e in (1, 2, 4, 8)
            for m in (128, 512)
            for n, k in ((256, 256), (1024, 1024), (2048, 1024), (1024, 2048))
        ]
    if shape_preset == "llama4":
        m_total = 16640
        return [
            ExperimentConfig(
                f"llama4_e{e}_m{m_total // e}_n{n}_k{k}",
                e,
                m_total // e,
                n,
                k,
            )
            for e in (1, 2, 4, 8)
            for n in (2048, 5120, 8192)
            for k in (2048, 5120, 8192)
        ]
    if shape_preset == "dsv3_671b":
        return [
            ExperimentConfig(
                "dsv3_671b_fwd_e8_m16384_n2048_k7168",
                8,
                16384,
                2048,
                7168,
            ),
            ExperimentConfig(
                "dsv3_671b_dgrad_e8_m16384_n7168_k2048",
                8,
                16384,
                7168,
                2048,
            ),
        ]
    raise ValueError(
        f"unsupported shape preset '{shape_preset}', expected smoke, sweep, "
        "llama4, or dsv3_671b"
    )


def _get_configs(args: argparse.Namespace) -> list[ExperimentConfig]:
    configs = (
        [_parse_shape(shape) for shape in args.shape]
        if args.shape
        else _preset_configs(args.shape_preset)
    )
    if args.n_limit is not None:
        configs = configs[: args.n_limit]
    for config in configs:
        _validate_config(config)
    return configs


def _make_column_major_weight_t(e: int, n: int, k: int) -> torch.Tensor:
    weight = torch.randn(e, n, k, dtype=torch.bfloat16, device="cuda")
    return weight.contiguous().transpose(-2, -1)


def _make_equal_group_offsets(config: ExperimentConfig) -> torch.Tensor:
    return torch.arange(
        config.m_per_group,
        (config.e + 1) * config.m_per_group,
        config.m_per_group,
        device="cuda",
        dtype=torch.int32,
    )


def _flops(config: ExperimentConfig) -> int:
    return 2 * config.m_total * config.n * config.k


def _logical_io_bytes(config: ExperimentConfig, out_dtype: torch.dtype) -> int:
    k_blocks = ceil_div(config.k, BLOCK_SIZE)
    n_blocks = ceil_div(config.n, BLOCK_SIZE)
    a_bytes = config.m_total * config.k
    b_bytes = config.e * config.k * config.n
    a_scale_bytes = config.m_total * k_blocks * 4
    b_scale_bytes = config.e * k_blocks * n_blocks * 4
    out_bytes = config.m_total * config.n * _dtype_size(out_dtype)
    return a_bytes + b_bytes + a_scale_bytes + b_scale_bytes + out_bytes


def _roofline_estimate(
    config: ExperimentConfig,
    out_dtype: torch.dtype,
    roofline_gpu_name: str,
) -> RooflineEstimate:
    specs = gpu_name_to_specs[roofline_gpu_name]
    achievable_gemm_pct = specs.get("pct_achievable_gemm_tops", 1.0)
    achievable_mem_pct = specs.get("pct_achievable_mem_bw", 1.0)

    flops = _flops(config)
    logical_io_bytes = _logical_io_bytes(config, out_dtype)
    fp8_compute_s = flops / (specs["fp8_peak_tops"] * achievable_gemm_pct)
    bf16_compute_s = flops / (specs["bf16_peak_tops"] * achievable_gemm_pct)
    fp8_memory_s = logical_io_bytes / (
        specs["peak_mem_bw_bytes_sec"] * achievable_mem_pct
    )
    launch_s = KERNEL_LAUNCH_OVERHEAD_SEC
    roofline_s = max(fp8_compute_s, fp8_memory_s, launch_s)
    if roofline_s == fp8_compute_s:
        bound = "compute"
    elif roofline_s == fp8_memory_s:
        bound = "memory"
    else:
        bound = "launch"

    return RooflineEstimate(
        fp8_compute_us=fp8_compute_s * 1e6,
        fp8_memory_us=fp8_memory_s * 1e6,
        fp8_launch_us=launch_s * 1e6,
        fp8_roofline_us=roofline_s * 1e6,
        fp8_roofline_bound=bound,
        fp8_roofline_tflops=(flops / 1e12) / roofline_s,
        bf16_compute_us=bf16_compute_s * 1e6,
        logical_io_bytes=logical_io_bytes,
        arithmetic_intensity_flops_per_byte=flops / logical_io_bytes,
    )


def _median_cuda_event_time_us(
    fn,
    *fn_args,
    warmup: int,
    iterations: int,
    rounds: int,
    **fn_kwargs,
) -> float:
    for _ in range(warmup):
        fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()

    times_us = []
    for _ in range(rounds):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(iterations):
            fn(*fn_args, **fn_kwargs)
        end_event.record()
        torch.cuda.synchronize()
        times_us.append(start_event.elapsed_time(end_event) * 1000.0 / iterations)
    return statistics.median(times_us)


def _profile_cuda_launches(
    fn,
    *fn_args,
    warmup: int,
    iterations: int,
    **fn_kwargs,
) -> None:
    for _ in range(warmup):
        fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    try:
        for _ in range(iterations):
            fn(*fn_args, **fn_kwargs)
        torch.cuda.synchronize()
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()


def _tflops(flops: int, us: Optional[float]) -> Optional[float]:
    if us is None or us <= 0:
        return None
    return (flops / 1e12) / (us / 1e6)


def _round_optional(value: Optional[float], ndigits: int = 3) -> Optional[float]:
    if value is None:
        return None
    return round(value, ndigits)


def _run_config(
    config: ExperimentConfig,
    out_dtype: torch.dtype,
    roofline_gpu_name: str,
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(args.seed)
    a = torch.randn(
        config.m_total,
        config.k,
        dtype=torch.bfloat16,
        device="cuda",
    )
    b_t = _make_column_major_weight_t(config.e, config.n, config.k)
    offs = _make_equal_group_offsets(config)

    a_fp8, a_scale = triton_fp8_blockwise_act_quant_lhs(a, dtype=torch.float8_e4m3fn)
    b_fp8, b_scale = triton_fp8_blockwise_weight_quant_grouped_forward_rhs(
        b_t,
        dtype=torch.float8_e4m3fn,
    )
    b_t_fp8 = b_fp8.transpose(-2, -1)
    b_t_scale = b_scale.transpose(-2, -1)

    flops = _flops(config)
    roofline = _roofline_estimate(config, out_dtype, roofline_gpu_name)

    bf16_us = None
    if not args.skip_bf16:
        bf16_us = _median_cuda_event_time_us(
            torch._grouped_mm,
            a,
            b_t,
            offs,
            out_dtype=out_dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )

    emulated_us = None
    if not args.skip_emulated:
        emulated_us = _median_cuda_event_time_us(
            emulated_blockwise_scaled_grouped_mm,
            a_fp8,
            b_t_fp8,
            a_scale,
            _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
            b_t_scale,
            _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
            offs,
            out_dtype,
            BLOCK_SIZE,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )

    cutedsl_us = None
    max_abs_diff = None
    if not args.skip_cutedsl:
        cutedsl_us = _median_cuda_event_time_us(
            cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d,
            a_fp8,
            b_t_fp8,
            a_scale,
            b_t_scale,
            offs,
            out_dtype,
            BLOCK_SIZE,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        if args.check_correctness:
            cutedsl_out = cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d(
                a_fp8,
                b_t_fp8,
                a_scale,
                b_t_scale,
                offs,
                out_dtype,
                BLOCK_SIZE,
            )
            ref = emulated_blockwise_scaled_grouped_mm(
                a_fp8,
                b_t_fp8,
                a_scale,
                _scaling_type_value(BLOCKWISE_1X128_SCALING_TYPE),
                b_t_scale,
                _scaling_type_value(BLOCKWISE_128X128_SCALING_TYPE),
                offs,
                out_dtype,
                BLOCK_SIZE,
            )
            max_abs_diff = (cutedsl_out.float() - ref.float()).abs().max().item()
            if not math.isfinite(max_abs_diff) or max_abs_diff > args.correctness_atol:
                raise AssertionError(
                    f"{config.name}: max_abs_diff={max_abs_diff} exceeds "
                    f"--correctness-atol={args.correctness_atol}"
                )
        if args.profile_cutedsl:
            print(
                f"profiling {config.name}: "
                f"warmup={args.profile_warmup} iterations={args.profile_iterations}"
            )
            _profile_cuda_launches(
                cutedsl_fp8_blockwise_scaled_grouped_mm_2d_3d,
                a_fp8,
                b_t_fp8,
                a_scale,
                b_t_scale,
                offs,
                out_dtype,
                BLOCK_SIZE,
                warmup=args.profile_warmup,
                iterations=args.profile_iterations,
            )

    cutedsl_tflops = _tflops(flops, cutedsl_us)
    emulated_tflops = _tflops(flops, emulated_us)
    bf16_tflops = _tflops(flops, bf16_us)
    cutedsl_roofline_pct = (
        None if cutedsl_us is None else (roofline.fp8_roofline_us / cutedsl_us) * 100.0
    )
    cutedsl_speedup_vs_emulated = (
        None if cutedsl_us is None or emulated_us is None else emulated_us / cutedsl_us
    )
    cutedsl_speedup_vs_bf16 = (
        None if cutedsl_us is None or bf16_us is None else bf16_us / cutedsl_us
    )

    return {
        "name": config.name,
        "E": config.e,
        "M_per_group": config.m_per_group,
        "M_total": config.m_total,
        "N": config.n,
        "K": config.k,
        "out_dtype": str(out_dtype).removeprefix("torch."),
        "flops": flops,
        "logical_io_bytes": roofline.logical_io_bytes,
        "arithmetic_intensity_flops_per_byte": round(
            roofline.arithmetic_intensity_flops_per_byte, 3
        ),
        "fp8_compute_roofline_us": round(roofline.fp8_compute_us, 3),
        "fp8_memory_roofline_us": round(roofline.fp8_memory_us, 3),
        "fp8_launch_roofline_us": round(roofline.fp8_launch_us, 3),
        "fp8_roofline_us": round(roofline.fp8_roofline_us, 3),
        "fp8_roofline_bound": roofline.fp8_roofline_bound,
        "fp8_roofline_tflops": round(roofline.fp8_roofline_tflops, 3),
        "bf16_compute_roofline_us": round(roofline.bf16_compute_us, 3),
        "bf16_us": _round_optional(bf16_us),
        "emulated_us": _round_optional(emulated_us),
        "cutedsl_us": _round_optional(cutedsl_us),
        "bf16_tflops": _round_optional(bf16_tflops),
        "emulated_tflops": _round_optional(emulated_tflops),
        "cutedsl_tflops": _round_optional(cutedsl_tflops),
        "cutedsl_roofline_pct": _round_optional(cutedsl_roofline_pct),
        "cutedsl_speedup_vs_emulated": _round_optional(cutedsl_speedup_vs_emulated),
        "cutedsl_speedup_vs_bf16": _round_optional(cutedsl_speedup_vs_bf16),
        "max_abs_diff_vs_emulated": _round_optional(max_abs_diff),
    }


def _write_csv(outfile: str, rows: Iterable[dict]):
    rows = list(rows)
    if not rows:
        return
    with open(outfile, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_results(rows: list[dict]):
    display_columns = [
        "name",
        "E",
        "M_per_group",
        "N",
        "K",
        "bf16_us",
        "emulated_us",
        "cutedsl_us",
        "cutedsl_tflops",
        "fp8_roofline_us",
        "fp8_roofline_bound",
        "cutedsl_roofline_pct",
        "cutedsl_speedup_vs_emulated",
    ]
    display_rows = [
        {column: row[column] for column in display_columns if column in row}
        for row in rows
    ]
    print(tabulate(display_rows, headers="keys", tablefmt="github"))


def run(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    if torch.version.hip:
        raise RuntimeError("CuTeDSL FP8 blockwise grouped GEMM benchmark is CUDA-only")
    if not is_sm_at_least_90():
        raise RuntimeError("FP8 blockwise grouped GEMM requires CUDA SM90+")
    if args.iterations <= 0 or args.warmup < 0 or args.rounds <= 0:
        raise ValueError("--iterations and --rounds must be positive; --warmup >= 0")
    if args.profile_cutedsl:
        if args.skip_cutedsl:
            raise ValueError("--profile-cutedsl requires the CuTeDSL path")
        if args.profile_iterations <= 0 or args.profile_warmup < 0:
            raise ValueError(
                "--profile-iterations must be positive; --profile-warmup >= 0"
            )
    if not args.skip_cutedsl and not _cutedsl_runtime_available():
        missing = ", ".join(_missing_cutedsl_runtime_packages())
        print(
            f"CuTeDSL runtime packages are not available ({missing}); "
            "skipping CuTeDSL path."
        )
        args.skip_cutedsl = True

    torch.random.manual_seed(args.seed)
    out_dtype = _dtype_from_name(args.out_dtype)
    roofline_gpu_name = get_roofline_gpu_name(args.roofline_gpu_name)
    configs = _get_configs(args)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch version: {torch.__version__}")
    print(f"roofline_gpu_name: {roofline_gpu_name}")
    print(f"out_dtype: {out_dtype}")
    print(f"warmup: {args.warmup}")
    print(f"iterations: {args.iterations}")
    print(f"rounds: {args.rounds}")
    print(f"shape_count: {len(configs)}")

    rows = []
    for config in configs:
        print(f"running {config.name}")
        rows.append(_run_config(config, out_dtype, roofline_gpu_name, args))

    _print_results(rows)
    if args.outfile:
        _write_csv(args.outfile, rows)
        print(f"wrote {args.outfile}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the CuTeDSL FP8 blockwise 2D x 3D grouped GEMM kernel "
            "against BF16, the emulated blockwise path, and an FP8 roofline target."
        )
    )
    parser.add_argument("--outfile", default=None)
    parser.add_argument(
        "--shape-preset",
        default="smoke",
        choices=("smoke", "sweep", "llama4", "dsv3_671b"),
        help="Preset shape set to run when --shape is not provided.",
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=None,
        help=(
            "Custom shape as 'E,M_PER_GROUP,N,K' or 'NAME,E,M_PER_GROUP,N,K'. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument("--n-limit", type=int, default=None)
    parser.add_argument(
        "--out-dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
    )
    parser.add_argument("--roofline-gpu-name", default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--skip-emulated", action="store_true")
    parser.add_argument("--skip-cutedsl", action="store_true")
    parser.add_argument("--check-correctness", action="store_true")
    parser.add_argument("--correctness-atol", type=float, default=2.0)
    parser.add_argument(
        "--profile-cutedsl",
        action="store_true",
        help=(
            "After timing/correctness, bracket warmed CuTeDSL launch(es) with "
            "cudaProfilerStart/Stop for ncu --profile-from-start off."
        ),
    )
    parser.add_argument("--profile-warmup", type=int, default=1)
    parser.add_argument("--profile-iterations", type=int, default=1)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

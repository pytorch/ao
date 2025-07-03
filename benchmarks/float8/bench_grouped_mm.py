# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import fire
import pandas as pd
import torch
import torch.utils.benchmark as benchmark
from utils import (
    get_gpu_kernel_gemm_time_s,
)

from torchao.testing.training.roofline_utils import get_specs


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean


def do_benchmarks(
    tops,
    peak_tops,
    use_gpu_kernel_time,
    f,
    *args,
    **kwargs,
):
    if use_gpu_kernel_time:
        # just the gemm GPU kernel
        time_sec = get_gpu_kernel_gemm_time_s(f, *args, **kwargs)
    else:
        # e2e time including kernel launch overhead
        time_sec = benchmark_fn_in_sec(f, *args, **kwargs)
    tops_sec = float(tops) / time_sec
    pct_top_peak = tops_sec / peak_tops
    return time_sec, tops_sec, pct_top_peak


@torch.inference_mode()
def run(
    n_limit: Optional[int] = None,
    out_filename: Optional[str] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    E: Optional[int] = None,  # dim 0 of B tensor (num experts)
    use_gpu_kernel_time: bool = True,
    shape_gen_name="llama4_17bx16e",
    recipe: str = "rowwise",
):
    device = "cuda"

    assert recipe in ("rowwise",), "unsupported"

    specs = get_specs()
    bf16_peak_tops = specs["bf16_peak_tops"]
    fp8_peak_tops = specs["fp8_peak_tops"]
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(f"peak tops: bf16 {bf16_peak_tops:.2e}, fp8 {fp8_peak_tops:.2e}")
    headers = (
        "name",
        "M",
        "K",
        "N",
        "E",
        "time_s",
        "speedup",
        "fp8_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N, E)

    for idx, (name, (M, K, N, E)) in enumerate(
        name_to_shapes,
    ):
        if n_limit is not None and idx >= n_limit:
            break
        assert M % E == 0, (
            "tokens (M) must be evenly divisible by num experts (E) for this benchmark"
        )
        tops = 2 * M * N * K * E
        print("M, K, N, E:", M, K, N, E, f"tops: {tops:.2E}")

        # Run bf16 torch._grouped_mm baseline.
        A = torch.randn(M, K, device=device, dtype=dtype)
        B = torch.randn(E, K, N, device=device, dtype=dtype)
        group_size = M // E
        offs = torch.arange(
            group_size, M + 1, group_size, dtype=torch.int32, device=device
        )
        ref_time_sec, ref_tops_sec, ref_pct_top_peak = do_benchmarks(
            tops,
            bf16_peak_tops,
            use_gpu_kernel_time,
            torch._grouped_mm,
            A,
            B,
            offs,
        )
        print(
            f"{dtype} time_sec {ref_time_sec:.2E}, tops/sec {ref_tops_sec:.2E}, pct_peak {ref_pct_top_peak:.3f}"
        )
        del A
        del B

        # Run scaled_grouped_mm.
        A_hp = torch.randn(M, K, device=device)
        B_hp_t = (
            torch.randn(E, K, N, device=device)
            .transpose(-2, -1)
            .contiguous()
            .transpose(-2, -1)
        )

        if recipe == "rowwise":
            # TODO: add e5m2
            A = A_hp.to(torch.float8_e4m3fn)
            B = B_hp_t.to(torch.float8_e4m3fn)
            peak_tops = fp8_peak_tops
            scale_a = torch.ones(M, device=device)
            scale_b = torch.ones(E, N, device=device)
        else:
            assert False, f"unknown recipe {recipe}"

        def do_scaled_grouped_mm(A, B):
            nonlocal scale_a
            nonlocal scale_b
            nonlocal offs
            return torch._scaled_grouped_mm(A, B, scale_a, scale_b, offs=offs)

        if recipe == "rowwise":
            do_matmul = do_scaled_grouped_mm
        else:
            raise ValueError(f"unknown recipe {recipe}")

        time_sec, tops_sec, pct_top_peak = do_benchmarks(
            tops, peak_tops, use_gpu_kernel_time, do_matmul, A, B
        )
        print(
            f"time_sec {time_sec:.2E}, tops/sec {tops_sec:.2E}, pct_peak {pct_top_peak:.3f}"
        )

        del A, B
        if scale_a is not None:
            del scale_a
        if scale_b is not None:
            del scale_b

        results.append(
            [
                name,
                M,
                K,
                N,
                E,
                ref_time_sec,
                time_sec,
                ref_time_sec / time_sec,
            ]
        )

    data_df = pd.DataFrame(results, columns=headers)
    print(data_df)

    if out_filename is not None:
        data_df.to_csv(out_filename)


def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    E: Optional[int] = None,
):
    M = 8192 if M is None else M
    if shape_gen_name == "llama4_17bx16e":
        # num_experts=16, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (M, 5120, 8192, 16),
            "moe.experts.w2": (M, 8192, 5120, 16),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "llama4_17bx128e":
        # num_experts=128, dim=5120
        names_to_shapes = {
            # M, K, N, E
            "moe.experts.w1": (M, 5120, 8192, 128),
            "moe.experts.w2": (M, 8192, 5120, 128),
        }
        return names_to_shapes.items()
    elif shape_gen_name == "custom":
        assert M is not None and K is not None and N is not None and E is not None, (
            "M, K, N, E must be specified for custom shape_gen"
        )
        name_to_shapes = {
            1: (M, K, N, E),
        }
        return name_to_shapes.items()

    raise AssertionError(f"unknown shape_gen_name {shape_gen_name}")


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()  # pragma: no cover

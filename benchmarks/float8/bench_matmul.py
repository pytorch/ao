# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from typing import Optional

import fire
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
from utils import (
    get_gpu_kernel_gemm_time_s,
    get_name_to_shapes_iter,
)

from torchao.testing.float8.roofline_utils import get_specs


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
    shape_gen_name: str = "pow2_extended",
    out_filename: Optional[str] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    use_gpu_kernel_time: bool = True,
    recipe: str = "tensorwise",
):
    device = "cuda"
    # TODO(future PR): this is ugly
    assert recipe in (
        "tensorwise",
        "rowwise",
        "mxfp8_cublas",
        "deepgemm_128_1_128_128",
    ), "unsupported"

    specs = get_specs()
    bf16_peak_tops = specs["bf16_peak_tops"]
    fp8_peak_tops = specs["fp8_peak_tops"]
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(f"peak tops: bf16 {bf16_peak_tops:.2e}, fp8 {fp8_peak_tops:.2e}")
    # TODO(this PR): make gpu kernel time work with deepgemm kernel
    print(f"use_gpu_kernel_time: {use_gpu_kernel_time}")
    print(f"recipe: {recipe}")

    headers = (
        "fast_accum",
        "name",
        "M",
        "K",
        "N",
        "ref_time_s",
        "fp8_time_s",
        "fp8_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    fast_accum_vals = [True, False]

    for idx, (fast_accum, (name, (M, K, N))) in enumerate(
        itertools.product(fast_accum_vals, name_to_shapes)
    ):
        if n_limit is not None and idx >= n_limit:
            break

        tops = 2 * M * N * K
        print("M, K, N:", M, K, N, f"tops: {tops:.2E}")

        # raw torch.mm
        A = torch.randn(M, K, device=device, dtype=dtype)
        m_ref = nn.Sequential(nn.Linear(K, N, dtype=dtype, device=device, bias=False))
        ref_time_sec, ref_tops_sec, ref_pct_top_peak = do_benchmarks(
            tops, bf16_peak_tops, use_gpu_kernel_time, m_ref, A
        )
        print(
            f"{dtype} time_sec {ref_time_sec:.2E}, tops/sec {ref_tops_sec:.2E}, pct_peak {ref_pct_top_peak:.3f}"
        )

        del A

        # raw float8 matmul (upper bound for what we can achive in eager mode)
        # TODO(future): add e5m2
        d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, dtype
        A = torch.zeros(M, K, device=device, dtype=d1)
        B = torch.zeros(K, N, device=device, dtype=d2).t().contiguous().t()
        if recipe == "tensorwise":
            scale_a = torch.tensor([1.0], device=device)
            scale_b = torch.tensor([1.0], device=device)
        elif recipe == "rowwise":
            scale_a = torch.ones(M, 1, device=device)
            scale_b = torch.ones(1, N, device=device)
        elif recipe == "mxfp8_cublas":
            scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
            scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        elif recipe == "deepgemm_128_1_128_128":
            scale_a = torch.ones(M, K // 128, device=device)
            scale_b = torch.ones(N // 128, K // 128, device=device)
        else:
            assert False, f"unknown recipe {recipe}"

        if recipe == "deepgemm_128_1_128_128":
            from torchao.prototype.deep_gemm_float8_training.deep_gemm_utils import (
                scaled_mm_deep_gemm_128_1_128_128,
            )

            def do_matmul(A, B):
                nonlocal scale_a
                nonlocal scale_b
                return scaled_mm_deep_gemm_128_1_128_128(A, B.t(), scale_a, scale_b)

        else:

            def do_matmul(A, B):
                nonlocal scale_a
                nonlocal scale_b
                return torch._scaled_mm(
                    A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
                )

        fp8_time_sec, fp8_tops_sec, fp8_pct_top_peak = do_benchmarks(
            tops, fp8_peak_tops, use_gpu_kernel_time, do_matmul, A, B
        )
        print(
            f"fp8 time_sec {fp8_time_sec:.2E}, tops/sec {fp8_tops_sec:.2E}, pct_peak {fp8_pct_top_peak:.3f}"
        )

        del A, B, scale_a, scale_b

        results.append(
            [
                fast_accum,
                name,
                M,
                K,
                N,
                ref_time_sec,
                fp8_time_sec,
                ref_time_sec / fp8_time_sec,
            ]
        )

    data_df = pd.DataFrame(results, columns=headers)
    print(data_df)

    if out_filename is not None:
        data_df.to_csv(out_filename)


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()  # pragma: no cover

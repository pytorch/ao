# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import itertools
from enum import IntEnum
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

from torchao.float8.config import ScalingGranularity

# estimating TOPs for matmuls in fp32, fp16, fp8
# assuming A * B = C, with A being M * K, B being K * N, C being M * N

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 989e12
h100_peak_tops_float8_tc = 1979e12

# HGX B20 specs: https://www.nvidia.com/en-us/data-center/hgx/
# note: divided numbers from ^ by 2 to undo the effects of sparsity
# TODO(this PR): I'm achieving 5% of peak TFLOPS with bf16 and float8,
# something seems funky
b200_peak_flops_float32 = 600e12
b200_peak_flops_fp16_tc = 18e15
b200_peak_tops_float8_tc = 36e15
b200_peak_tops_float4_tc = 72e15

dtype_to_peak_tops_h100 = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}

dtype_to_peak_tops_b200 = {
    torch.float32: b200_peak_flops_float32,
    torch.float16: b200_peak_flops_fp16_tc,
    torch.bfloat16: b200_peak_flops_fp16_tc,
    torch.float8_e4m3fn: b200_peak_tops_float8_tc,
    torch.float8_e5m2: b200_peak_tops_float8_tc,
    # TODO float4
}

# TODO(this PR): switch automatically by detected hardware type
# TODO(this PR): fp4 is currently using fp8's peak tops below, fix it
dtype_to_peak_tops = dtype_to_peak_tops_b200


# not for land, matching https://www.internalfb.com/phabricator/paste/view/P1717686991
class DataType(IntEnum):
    DEFAULT = 0
    E8M0 = 1
    FP4 = 2
    UFP8 = 3


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
    shape_gen_name: str = "llama",
    out_filename: Optional[str] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
    use_gpu_kernel_time: bool = False,
    scaling_granularity: str = "tensorwise",
    blockwise_dtype: Optional[str] = None,
):
    device = "cuda"

    headers = (
        "fast_accum",
        "name",
        "M",
        "K",
        "N",
        "ref_time_s",
        "lowp_time_s",
        "lowp_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    fast_accum_vals = [True, False]
    # Note: blockwise not in enum because blockwise is in prototype
    if scaling_granularity != "blockwise":
        scaling_granularity = ScalingGranularity(scaling_granularity)

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
            tops, dtype_to_peak_tops[dtype], use_gpu_kernel_time, m_ref, A
        )
        print(
            f"{dtype} time_sec {ref_time_sec:.2E}, tops/sec {ref_tops_sec:.2E}, pct_peak {ref_pct_top_peak:.3f}"
        )

        del A

        # raw float8 matmul (upper bound for what we can achive in eager mode)
        # TODO(future): add e5m2
        d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, dtype
        A = torch.randn(M, K, device=device).to(d1)
        B = torch.randn(K, N, device=device).to(d2).t().contiguous().t()
        if scaling_granularity == ScalingGranularity.TENSORWISE:
            scale_a = torch.tensor([1.0], device=device)
            scale_b = torch.tensor([1.0], device=device)
        elif scaling_granularity == ScalingGranularity.AXISWISE:
            scale_a = torch.ones(M, 1, device=device)
            scale_b = torch.ones(1, N, device=device)
        elif scaling_granularity == "blockwise" and blockwise_dtype == "float8_e4m3":
            # TODO(this PR): also block size 16
            BLOCK_SIZE = 32
            A = torch.randint(128, (M, K), device=device, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            B = (
                torch.randint(128, (N, K), device=device, dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .t()
            )
            scale_a = torch.randint(
                128, (M, K // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
            )
            scale_b = torch.randint(
                128, (N, K // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
            ).t()
        elif scaling_granularity == "blockwise" and blockwise_dtype == "float4":
            # TODO(this PR): also block size 16
            BLOCK_SIZE = 16
            A = torch.randint(128, (M, K // 2), device=device, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            B = (
                torch.randint(128, (N, K // 2), device=device, dtype=torch.uint8)
                .view(torch.float8_e4m3fn)
                .t()
            )
            scale_a = torch.randint(
                128, (M, K // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
            )
            scale_b = torch.randint(
                128, (N, K // BLOCK_SIZE), dtype=torch.uint8, device="cuda"
            ).t()
        else:
            raise AssertionError(f"unsupported granularity {scaling_granularity}")

        def do_matmul(A, B):
            nonlocal scale_a
            nonlocal scale_b

            if scaling_granularity == "blockwise" and blockwise_dtype == "float8_e4m3":
                return torch._scaled_mm(
                    A,
                    B,
                    scale_a,
                    scale_b,
                    bias=None,
                    scale_result=None,
                    out_dtype=d3,
                    use_fast_accum=fast_accum,
                    a_dtype=None,  # inferred from A
                    b_dtype=None,  # inferred from B
                    scale_dtype=DataType.E8M0,
                )
            elif scaling_granularity == "blockwise" and blockwise_dtype == "float4":
                return torch._scaled_mm(
                    A,
                    B,
                    scale_a,
                    scale_b,
                    bias=None,
                    scale_result=None,
                    out_dtype=d3,
                    use_fast_accum=fast_accum,
                    a_dtype=DataType.FP4,
                    b_dtype=DataType.FP4,
                    scale_dtype=DataType.E8M0,
                )

            else:
                return torch._scaled_mm(
                    A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
                )

        # test
        # res = do_matmul(A, B)

        fp8_time_sec, fp8_tops_sec, fp8_pct_top_peak = do_benchmarks(
            tops, dtype_to_peak_tops[d1], use_gpu_kernel_time, do_matmul, A, B
        )
        print(
            f"lowp time_sec {fp8_time_sec:.2E}, tops/sec {fp8_tops_sec:.2E}, pct_peak {fp8_pct_top_peak:.3f}"
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

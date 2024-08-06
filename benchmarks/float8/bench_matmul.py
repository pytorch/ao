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

# estimating TOPs for matmuls in fp32, fp16, fp8
# assuming A * B = C, with A being M * K, B being K * N, C being M * N

# H100 SXM specs: bottom of https://www.nvidia.com/en-us/data-center/h100/
h100_peak_flops_float32 = 67e12
h100_peak_flops_fp16_tc = 989e12
h100_peak_tops_float8_tc = 1979e12

dtype_to_peak_tops = {
    torch.float32: h100_peak_flops_float32,
    torch.float16: h100_peak_flops_fp16_tc,
    torch.bfloat16: h100_peak_flops_fp16_tc,
    torch.float8_e4m3fn: h100_peak_tops_float8_tc,
    torch.float8_e5m2: h100_peak_tops_float8_tc,
}


def benchmark_fn_in_sec(f, *args, **kwargs):
    # Manual warmup
    for _ in range(4):
        f(*args, **kwargs)
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    measurement = t0.blocked_autorange()
    return measurement.mean


def do_benchmarks(tops, peak_tops, f, *args, **kwargs):
    time_sec = benchmark_fn_in_sec(f, *args, **kwargs)
    tops_sec = float(tops) / time_sec
    pct_top_peak = tops_sec / peak_tops
    return time_sec, tops_sec, pct_top_peak


def get_name_to_shapes_iter(
    shape_gen_name: str,
    M: Optional[int],
    K: Optional[int],
    N: Optional[int],
):
    if shape_gen_name == 'llama':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        bsz, seq_len = 4, 4096
        M = bsz * seq_len
        # LLaMa 2 70B single-node weight shapes
        # assumes fused attn.wqkv and ffn.w13
        # source: https://fburl.com/gsheet/g8onr7rh
        name_to_shapes_70b = {
            "attn.wqkv": (M, 8192, 1280),
            "attn.w0": (M, 1024, 8192),
            "ffn.w13": (M, 8192, 7168),
            "ffn.w2": (M, 3584, 8192),
        }
        return name_to_shapes_70b.items()

    elif shape_gen_name == 'square':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_power_of_2 = 5  # 32
        max_power_of_2 = 16  # 65,536
        for idx, power_of_2 in enumerate(range(min_power_of_2, max_power_of_2 + 1)):
            val = 2 ** power_of_2
            name_to_shapes[idx] = val, val, val
        return name_to_shapes.items()

    elif shape_gen_name == 'sweep':
        assert M == K == N == None, \
            f'M, K, N arguments not supported for shape_gen_name {shape_gen_name}'
        name_to_shapes = {}
        min_p2 = 5  # 32
        max_p2 = 16  # 65,536
        counter = 0
        for M_p2 in range(min_p2, max_p2 + 1):
            M = 2 ** M_p2
            for K_p2 in range(min_p2, max_p2 + 1):
                K = 2 ** K_p2
                for N_p2 in range(min_p2, max_p2 + 1):
                    N = 2 ** N_p2
                    name_to_shapes[counter] = M, K, N
                    counter += 1
        return name_to_shapes.items()

    elif shape_gen_name == 'custom':
        assert M is not None and K is not None and N is not None, \
            'M, K, N must be specified for custom shape_gen'
        name_to_shapes = {
            1: (M, K, N),
        }
        return name_to_shapes.items()

    raise AssertionError(f'unknown shape_gen_name {shape_gen_name}')


@torch.inference_mode()
def run(
    n_limit: Optional[int] = None,
    shape_gen_name: str = 'llama',
    out_filename: Optional[str] = None,
    M: Optional[int] = None,
    K: Optional[int] = None,
    N: Optional[int] = None,
):
    device = "cuda"

    headers = ("fast_accum", "name", "M", "K", "N", "ref_time_s", "fp8_time_s", "fp8_speedup")
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    fast_accum_vals = [True, False]

    for idx, (fast_accum, (name, (M, K, N))) in enumerate(itertools.product(fast_accum_vals, name_to_shapes)):
        if n_limit is not None and idx >= n_limit:
            break

        tops = 2 * M * N * K
        print("M, K, N:", M, K, N, f"tops: {tops:.2E}")

        # raw torch.mm
        A = torch.randn(M, K, device=device, dtype=dtype)
        m_ref = nn.Sequential(nn.Linear(K, N, dtype=dtype, device=device, bias=False))
        ref_time_sec, ref_tops_sec, ref_pct_top_peak = do_benchmarks(
            tops, dtype_to_peak_tops[dtype], m_ref, A
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
        scale_a = torch.tensor([1.0], device=device)
        scale_b = torch.tensor([1.0], device=device)

        def do_matmul(A, B):
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
            )

        fp8_time_sec, fp8_tops_sec, fp8_pct_top_peak = do_benchmarks(
            tops, dtype_to_peak_tops[d1], do_matmul, A, B
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

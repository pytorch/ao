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
from utils import (
    do_benchmarks,
    get_name_to_shapes_iter,
)

from torchao.ops import mx_fp4_bf16
from torchao.prototype.mx_formats.mx_tensor import to_mx
from torchao.testing.training.roofline_utils import get_specs


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
        "mxfp4_cutlass",
        "nvfp4",
    ), "unsupported"
    use_fp4 = recipe in ("mxfp4_cutlass", "nvfp4")

    specs = get_specs()
    bf16_peak_tops = specs["bf16_peak_tops"]
    fp8_peak_tops = specs["fp8_peak_tops"]
    fp4_peak_tops = specs.get("fp4_peak_tops", 0.0)  # only on sm120
    print(f"gpu_name: {torch.cuda.get_device_name(0)}")
    print(
        f"peak tops: bf16 {bf16_peak_tops:.2e}, fp8 {fp8_peak_tops:.2e}, fp4 {fp4_peak_tops:.2e}"
    )
    headers = (
        "fast_accum",
        "name",
        "M",
        "K",
        "N",
        "time_s",
        "speedup",
        "fp8_speedup",
    )
    results = []

    dtype = torch.bfloat16
    name_to_shapes = get_name_to_shapes_iter(shape_gen_name, M, K, N)
    fast_accum_vals = [False] if use_fp4 else [True, False]

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

        A_hp = torch.randn(M, K, device=device)
        B_hp_t = torch.randn(N, K, device=device)

        if recipe == "mxfp4_cutlass":
            _, A = to_mx(A_hp, torch.float4_e2m1fn_x2, 32)
            _, Bt = to_mx(B_hp_t, torch.float4_e2m1fn_x2, 32)
            B = Bt.contiguous().T
            peak_tops = fp4_peak_tops
        elif recipe == "nvfp4":
            from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_quantize

            A_scales, A_data = nvfp4_quantize(A_hp, block_size=16)
            B_scales, B_data = nvfp4_quantize(B_hp_t, block_size=16)
            A = A_data.view(torch.float4_e2m1fn_x2)
            B = B_data.view(torch.float4_e2m1fn_x2).T
            peak_tops = fp4_peak_tops
        else:
            # raw float8 matmul (upper bound for what we can achive in eager mode)
            # TODO(future): add e5m2
            d1, d2, d3 = torch.float8_e4m3fn, torch.float8_e4m3fn, dtype
            A = A_hp.to(d1)
            B = B_hp_t.to(d2).contiguous().T
            peak_tops = fp8_peak_tops

        if recipe == "tensorwise":
            scale_a = torch.tensor([1.0], device=device)
            scale_b = torch.tensor([1.0], device=device)
        elif recipe == "rowwise":
            scale_a = torch.ones(M, 1, device=device)
            scale_b = torch.ones(1, N, device=device)
        elif recipe in ("mxfp8_cublas", "mxfp4_cutlass"):
            scale_a = torch.ones(M, K // 32, device=device, dtype=torch.float8_e8m0fnu)
            scale_b = torch.ones(N, K // 32, device=device, dtype=torch.float8_e8m0fnu)
        elif recipe == "nvfp4":
            # Use the blockwise scales from nvfp4_quantize
            scale_a = A_scales.view(torch.float8_e4m3fn)
            scale_b = B_scales.view(torch.float8_e4m3fn)
        else:
            assert False, f"unknown recipe {recipe}"

        def do_matmul_fp8(A, B):
            nonlocal scale_a
            nonlocal scale_b
            return torch._scaled_mm(
                A, B, scale_a, scale_b, out_dtype=d3, use_fast_accum=fast_accum
            )

        def do_matmul_mxfp4(A, B):
            nonlocal scale_a
            nonlocal scale_b
            return mx_fp4_bf16(A, B, scale_a, scale_b)

        def do_matmul_nvfp4(A, B):
            nonlocal scale_a
            nonlocal scale_b
            return torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=dtype)

        def do_grouped_mm(A, B):
            return torch._grouped_mm(A, B, use_fast_accum=fast_accum)

        def do_scaled_grouped_mm(A, B):
            nonlocal scale_a
            nonlocal scale_b
            return torch._scaled_grouped_mm(
                A, B, scale_a, scale_b, use_fast_accum=fast_accum
            )

        if recipe == "mxfp4_cutlass":
            do_matmul = do_matmul_mxfp4
        elif recipe == "nvfp4":
            do_matmul = do_matmul_nvfp4
        else:
            do_matmul = do_matmul_fp8

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
                fast_accum,
                name,
                M,
                K,
                N,
                ref_time_sec,
                time_sec,
                ref_time_sec / time_sec,
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

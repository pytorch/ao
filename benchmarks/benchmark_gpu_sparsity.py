# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from torch.sparse import SparseSemiStructuredTensor, to_sparse_semi_structured
from torch.sparse._triton_ops_meta import optimize_bsr_dense_addmm
from tqdm import tqdm

from torchao.sparsity.utils import (
    create_block_sparse_tensor,
    create_semi_structured_tensor,
)
from torchao.utils import benchmark_model

torch.set_printoptions(
    precision=2,
    threshold=None,
    edgeitems=16,
    linewidth=480,
    profile=None,
    sci_mode=False,
)


def benchmark_model_with_warmup(func, x, N_WARMUP=3):
    benchmark_model(func, N_WARMUP, device_type="cuda")
    return benchmark_model(func, 10, device_type="cuda")


def run_gpu_sparse_benchmark(m, k, n, args):
    with torch.no_grad():
        dtype = getattr(torch, args.dtype)

        x = torch.randn(n, k).to(dtype).cuda()

        # handle sparsity types
        if args.sparsity == "semi-structured":
            SparseSemiStructuredTensor._FORCE_CUTLASS = args.backend == "cutlass"
            A = create_semi_structured_tensor(m, k, dtype)
            A_sparse = to_sparse_semi_structured(A)
        elif args.sparsity == "block-sparse":
            A = create_block_sparse_tensor(
                m, k, args.block_size, args.sparsity_level, dtype
            )
            A_sparse = A.to_sparse_bsr(blocksize=args.block_size)
            # BSR kernel tuning
            if args.bsr_autotune:
                print("Tuning kernel params")
                optimize_bsr_dense_addmm(
                    m,
                    k,
                    n,
                    args.block_size,
                    args.block_size,
                    dtype=dtype,
                    sparsity=args.sparsity_level,
                    verbose=True,
                )
        else:
            raise ValueError(f"Unknown sparsity: {args.sparsity}")

        if args.eval_fn == "linear":
            b = torch.randn(m, dtype=dtype).cuda()

            # can't use lambda
            def dense_func():
                return F.linear(x, A, b)

            def sparse_func():
                return F.linear(x, A_sparse, b)

        elif args.eval_fn == "mm":
            if dtype == torch.float8_e4m3fn:
                x = x.t()

                scale_a = torch.tensor([1.0], device="cuda")
                scale_b = torch.tensor([1.0], device="cuda")

                def dense_func():
                    return torch._scaled_mm(
                        A, x, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.bfloat16
                    )

                def sparse_func():
                    return torch._scaled_mm(
                        A_sparse,
                        x,
                        scale_a=scale_a,
                        scale_b=scale_b,
                        out_dtype=torch.bfloat16,
                    )
            else:
                x = x.t()

                def dense_func():
                    return torch.mm(A, x)

                def sparse_func():
                    return torch.mm(A_sparse, x)
        else:
            raise ValueError(f"Unknown eval_fn: {args.eval_fn}")

        dense_time = benchmark_model_with_warmup(dense_func, "dense.json.gz")
        sparse_time = benchmark_model_with_warmup(sparse_func, "sparse.json.gz")

        dense_func_c = torch.compile(dense_func, mode="max-autotune")
        dense_time_c = benchmark_model_with_warmup(
            dense_func_c, "dense_compile.json.gz"
        )

        sparse_func_c = torch.compile(sparse_func, mode="max-autotune")
        sparse_time_c = benchmark_model_with_warmup(
            sparse_func_c, "sparse_compile.json.gz"
        )

        torch._dynamo.reset()

        return {
            "test_function": args.eval_fn,
            "m": m,
            "k": k,
            "n": n,
            "dtype": args.dtype,
            "sparse": sparse_time,
            "dense": dense_time,
            "dense_c": dense_time_c,
            "sparse_c": sparse_time_c,
            "speedup (d/s)": min(dense_time, dense_time_c)
            / min(sparse_time, sparse_time_c),
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Sparsity Microbenchmarks")
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "llama3-8b-a",
            "llama3-8b-w",
            "vit-mlp",
            "nvidia-fixed-k",
            "nvidia-fixed-mn",
        ],
    )
    parser.add_argument(
        "--sparsity",
        type=str,
        choices=[
            "semi-structured",
            "block-sparse",
        ],
    )
    parser.add_argument(
        "--sparsity-level",
        type=float,
    )
    parser.add_argument(
        "--block-size",
        type=int,
        choices=[
            16,
            32,
            64,
        ],
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["int8", "float16", "bfloat16", "float32", "float8_e4m3fn"],
        default="bfloat16",
    )
    parser.add_argument(
        "--backend", type=str, choices=["cutlass", "cusparselt"], default="cusparselt"
    )
    parser.add_argument(
        "--eval-fn", type=str, choices=["linear", "mm"], default="linear"
    )
    parser.add_argument("-contiguous", action="store_true")
    parser.add_argument("-save", action="store_true")
    parser.add_argument(
        "-bsr-autotune", action="store_true", help="Tune BSR kernel parameters"
    )
    args = parser.parse_args()

    print(f"Started benchmark: {args}")

    if args.mode == "llama3-8b-a":
        mm_shapes = [
            (4096, 13312, 16384),
            (4096, 16384, 6560),
            (4096, 22528, 32768),
            (4096, 32768, 11264),
            (4096, 5632, 16384),
            (4096, 16384, 2816),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args) for (m, n, k) in tqdm(mm_shapes)
        )
    elif args.mode == "llama3-8b-w":
        mm_shapes = [
            (16, 4096, 11008),
            (16, 4096, 4096),
            (16, 11008, 4096),
            (4096, 4096, 11008),
            (4096, 4096, 4096),
            (4096, 11008, 4096),
            (8192, 4096, 11008),
            (8192, 4096, 4096),
            (8192, 11008, 4096),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args) for (m, k, n) in tqdm(mm_shapes)
        )
    elif args.mode == "vit-mlp":
        vit_shapes = [
            # vit-base
            (768, 3072, 50432),
            (3072, 3072, 50432),
            # vit-huge
            (1280, 5120, 65792),
            (5120, 1280, 65792),
        ]
        results = (
            run_gpu_sparse_benchmark(m, k, n, args) for (m, k, n) in tqdm(vit_shapes)
        )
    elif args.mode == "nvidia-fixed-k":
        mn_vals = [
            3072,
            4096,
            5120,
            6144,
            7168,
            8192,
            9216,
            10240,
            11264,
            12288,
            13312,
            14336,
            15360,
            16384,
            17408,
            18432,
            19456,
            20480,
        ]
        results = (
            run_gpu_sparse_benchmark(mn, 10240, mn, args) for mn in tqdm(mn_vals)
        )
    elif args.mode == "nvidia-fixed-mn":
        k_vals = [
            2560,
            3840,
            5120,
            6400,
            7680,
            8960,
            10240,
            11520,
            12800,
            14080,
            15360,
            16640,
            17920,
            19200,
            20480,
        ]
        results = (
            run_gpu_sparse_benchmark(10240, k, 10240, args) for k in tqdm(k_vals)
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    df = pd.DataFrame.from_records(results)
    if args.save:
        save_file = f"{args.mode}_{args.dtype}_{args.backend}.csv"
        df.to_csv(save_file)
        print(f"Finished benchmark: {args.mode} saved results to {save_file}")
    print(df)
